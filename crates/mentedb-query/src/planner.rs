//! Query planner: converts a parsed `Statement` into a `QueryPlan`.

use crate::ast::*;
use mentedb_core::edge::EdgeType;
use mentedb_core::error::{MenteError, MenteResult};
use mentedb_core::types::{MemoryId, Timestamp};
use serde::{Deserialize, Serialize};

/// A physical query plan describing how to execute a query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QueryPlan {
    VectorSearch {
        query: Vec<f32>,
        k: usize,
        filters: Vec<Filter>,
        /// Full boolean WHERE tree, when the clause used OR/NOT/grouping. When set,
        /// the executor post-filters candidates by this instead of `filters`.
        #[serde(default)]
        condition: Option<Condition>,
        /// ORDER BY clause; the executor sorts the result set by it before LIMIT.
        #[serde(default)]
        order_by: Option<OrderBy>,
    },
    TagScan {
        tags: Vec<String>,
        filters: Vec<Filter>,
        limit: Option<usize>,
        /// Full boolean WHERE tree, when the clause used OR/NOT/grouping. When set,
        /// the executor post-filters candidates by this instead of `filters`.
        #[serde(default)]
        condition: Option<Condition>,
        /// ORDER BY clause; the executor sorts the result set by it before LIMIT.
        #[serde(default)]
        order_by: Option<OrderBy>,
    },
    /// A scan narrowed by the temporal (creation time) index. `start..=end` is
    /// the creation range the index admits; `filters` holds every WHERE filter,
    /// the range ones included, and the executor re-checks them all on each
    /// loaded memory, so strict bounds (`>`, `<`) stay exact.
    TemporalScan {
        start: Timestamp,
        end: Timestamp,
        filters: Vec<Filter>,
        /// LIMIT; the executor falls back to its configured default when unset.
        #[serde(default)]
        limit: Option<usize>,
        /// ORDER BY clause; the executor sorts the result set by it before LIMIT.
        #[serde(default)]
        order_by: Option<OrderBy>,
    },
    GraphTraversal {
        start: MemoryId,
        depth: usize,
        edge_types: Vec<EdgeType>,
    },
    PointLookup {
        id: MemoryId,
    },
    EdgeInsert {
        source: MemoryId,
        target: MemoryId,
        edge_type: EdgeType,
        weight: f32,
    },
    Delete {
        id: MemoryId,
    },
    Consolidate {
        filters: Vec<Filter>,
    },
}

const DEFAULT_LIMIT: usize = 20;

/// Produce an execution plan from a parsed statement.
pub fn plan(statement: &Statement) -> MenteResult<QueryPlan> {
    match statement {
        Statement::Recall(recall) => {
            validate_filters(&recall.filters)?;
            if let Some(cond) = &recall.condition {
                validate_condition(cond)?;
            }
        }
        Statement::Consolidate(cons) => validate_filters(&cons.filters)?,
        _ => {}
    }
    match statement {
        Statement::Recall(recall) => plan_recall(recall),
        Statement::Relate(relate) => plan_relate(relate),
        Statement::Forget(forget) => Ok(QueryPlan::Delete { id: forget.target }),
        Statement::Consolidate(cons) => Ok(QueryPlan::Consolidate {
            filters: cons.filters.clone(),
        }),
        Statement::Traverse(trav) => plan_traverse(trav),
    }
}

fn plan_recall(recall: &RecallStatement) -> MenteResult<QueryPlan> {
    let limit = recall.limit.unwrap_or(DEFAULT_LIMIT);

    // If there's a NEAR clause or a SimilarTo filter, use vector search
    if let Some(ref vec) = recall.near {
        return Ok(QueryPlan::VectorSearch {
            query: vec.clone(),
            k: limit,
            filters: recall.filters.clone(),
            condition: recall.condition.clone(),
            order_by: recall.order_by.clone(),
        });
    }

    // OR/NOT/grouped WHERE: none of the leaf-based index optimizations below are
    // safe (they assume every filter must hold), so full-scan and let the executor
    // evaluate the boolean tree. Correct, if less selective, for these rarer queries.
    if recall.condition.is_some() {
        return Ok(QueryPlan::TagScan {
            tags: Vec::new(),
            filters: Vec::new(),
            limit: Some(limit),
            condition: recall.condition.clone(),
            order_by: recall.order_by.clone(),
        });
    }

    // Check for SimilarTo operator in filters — implies vector search via text embedding
    if let Some(sim_filter) = recall.filters.iter().find(|f| f.op == Operator::SimilarTo) {
        if let Value::Text(ref _text) = sim_filter.value {
            // The text is embedded at execution time (no embedder here), so keep
            // ALL filters, including the SimilarTo one: the executor reads its
            // text to build the query vector and applies the rest (type, tag).
            return Ok(QueryPlan::VectorSearch {
                query: Vec::new(), // placeholder — executor embeds the SimilarTo text
                k: limit,
                filters: recall.filters.clone(),
                condition: None,
                order_by: recall.order_by.clone(),
            });
        }
        // SimilarTo with non-text value doesn't make sense
        return Err(MenteError::Query(
            "~> operator requires a text value on the right-hand side".into(),
        ));
    }

    // If only tag filters, use TagScan
    let tag_filters: Vec<&Filter> = recall
        .filters
        .iter()
        .filter(|f| f.field == Field::Tag)
        .collect();
    if !tag_filters.is_empty() && recall.filters.iter().all(|f| f.field == Field::Tag) {
        let tags: Vec<String> = tag_filters
            .iter()
            .filter_map(|f| match &f.value {
                Value::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect();
        return Ok(QueryPlan::TagScan {
            tags,
            filters: Vec::new(),
            limit: Some(limit),
            condition: None,
            order_by: recall.order_by.clone(),
        });
    }

    // A numeric range on `created` narrows through the temporal index, which
    // is keyed by creation time. Ranges on `accessed` (or non-numeric bounds)
    // cannot use it and fall through to the filtered scan below.
    let mut start: Option<Timestamp> = None;
    let mut end: Option<Timestamp> = None;
    for f in &recall.filters {
        if f.field != Field::Created {
            continue;
        }
        let Some(ts) = timestamp_value(&f.value) else {
            continue;
        };
        match f.op {
            Operator::Gt | Operator::Gte => start = Some(start.map_or(ts, |s| s.max(ts))),
            Operator::Lt | Operator::Lte => end = Some(end.map_or(ts, |e| e.min(ts))),
            _ => {}
        }
    }
    if start.is_some() || end.is_some() {
        return Ok(QueryPlan::TemporalScan {
            start: start.unwrap_or(0),
            end: end.unwrap_or(Timestamp::MAX),
            // Every filter, the range ones too: the index bound is inclusive,
            // so the executor's re-check is what makes `>` and `<` strict.
            filters: recall.filters.clone(),
            limit: Some(limit),
            order_by: recall.order_by.clone(),
        });
    }

    // Fallback: tag scan with no tags (full scan with filters)
    Ok(QueryPlan::TagScan {
        tags: Vec::new(),
        filters: recall.filters.clone(),
        limit: Some(limit),
        condition: None,
        order_by: recall.order_by.clone(),
    })
}

/// A numeric literal as a timestamp, clamping negatives to zero.
fn timestamp_value(v: &Value) -> Option<Timestamp> {
    match v {
        Value::Integer(i) => Some((*i).max(0) as Timestamp),
        Value::Number(n) if n.is_finite() => Some(n.max(0.0) as Timestamp),
        _ => None,
    }
}

/// Reject WHERE leaves the executor cannot evaluate, so a malformed clause is a
/// query error instead of silently matching every memory.
pub fn validate_filters(filters: &[Filter]) -> MenteResult<()> {
    filters.iter().try_for_each(validate_filter)
}

/// [`validate_filters`] over every leaf of a boolean WHERE tree.
pub fn validate_condition(c: &Condition) -> MenteResult<()> {
    match c {
        Condition::Leaf(f) => validate_filter(f),
        Condition::And(children) | Condition::Or(children) => {
            children.iter().try_for_each(validate_condition)
        }
        Condition::Not(inner) => validate_condition(inner),
    }
}

/// Check one filter on the owner and score fields:
///
/// - `agent` and `space` compare an id (a UUID, quoted or bare) with `=`,
///   `!=`, or `IN [...]`.
/// - `salience` and `confidence` compare a number with `=`, `!=`, `<`, `<=`,
///   `>`, `>=`, or `IN [...]`. At execution `salience` is the memory's current
///   decayed salience, not the stored base; `confidence` is the stored value.
///
/// Other fields keep their existing handling.
pub fn validate_filter(f: &Filter) -> MenteResult<()> {
    let (name, id_field) = match f.field {
        Field::Agent => ("agent", true),
        Field::Space => ("space", true),
        Field::Salience => ("salience", false),
        Field::Confidence => ("confidence", false),
        _ => return Ok(()),
    };
    let op_ok = if id_field {
        matches!(f.op, Operator::Eq | Operator::Neq | Operator::In)
    } else {
        matches!(
            f.op,
            Operator::Eq
                | Operator::Neq
                | Operator::Lt
                | Operator::Lte
                | Operator::Gt
                | Operator::Gte
                | Operator::In
        )
    };
    if !op_ok {
        let allowed = if id_field {
            "=, != or IN"
        } else {
            "=, !=, <, <=, >, >= or IN"
        };
        return Err(MenteError::Query(format!(
            "{name} supports only {allowed}, found {:?}",
            f.op
        )));
    }
    let value_ok = |v: &Value| {
        if id_field {
            matches!(v, Value::Uuid(_))
        } else {
            matches!(v, Value::Number(_) | Value::Integer(_))
        }
    };
    let ok = match &f.value {
        Value::List(items) => f.op == Operator::In && items.iter().all(value_ok),
        v => f.op != Operator::In && value_ok(v),
    };
    if ok {
        Ok(())
    } else {
        let expected = if id_field { "a UUID" } else { "a number" };
        Err(MenteError::Query(format!(
            "{name} must be compared with {expected}, found {:?}",
            f.value
        )))
    }
}

fn plan_relate(relate: &RelateStatement) -> MenteResult<QueryPlan> {
    Ok(QueryPlan::EdgeInsert {
        source: relate.source,
        target: relate.target,
        edge_type: relate.edge_type,
        weight: relate.weight.unwrap_or(1.0),
    })
}

fn plan_traverse(trav: &TraverseStatement) -> MenteResult<QueryPlan> {
    Ok(QueryPlan::GraphTraversal {
        start: trav.start,
        depth: trav.depth,
        edge_types: trav.edge_filter.clone().unwrap_or_default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;
    use crate::parser::Parser;

    fn plan_mql(input: &str) -> QueryPlan {
        let tokens = tokenize(input).unwrap();
        let stmt = Parser::parse(&tokens).unwrap();
        plan(&stmt).unwrap()
    }

    #[test]
    fn test_near_produces_vector_search() {
        let qp = plan_mql("RECALL memories NEAR [0.1, 0.2, 0.3] LIMIT 5");
        match qp {
            QueryPlan::VectorSearch { query, k, .. } => {
                assert_eq!(query, vec![0.1, 0.2, 0.3]);
                assert_eq!(k, 5);
            }
            _ => panic!("expected VectorSearch, got {:?}", qp),
        }
    }

    #[test]
    fn test_similar_to_produces_vector_search() {
        let qp = plan_mql(r#"RECALL memories WHERE content ~> "database migration" LIMIT 10"#);
        match qp {
            QueryPlan::VectorSearch { k, .. } => {
                assert_eq!(k, 10);
            }
            _ => panic!("expected VectorSearch, got {:?}", qp),
        }
    }

    #[test]
    fn test_tag_filter_produces_tag_scan() {
        let qp = plan_mql(r#"RECALL memories WHERE tag = "backend" LIMIT 5"#);
        match qp {
            QueryPlan::TagScan { tags, limit, .. } => {
                assert_eq!(tags, vec!["backend".to_string()]);
                assert_eq!(limit, Some(5));
            }
            _ => panic!("expected TagScan, got {:?}", qp),
        }
    }

    #[test]
    fn test_created_range_produces_temporal_scan_with_every_filter() {
        let qp = plan_mql(
            "RECALL memories WHERE created > 100 AND created >= 50 AND created < 900 AND type = episodic ORDER BY salience DESC LIMIT 10",
        );
        match qp {
            QueryPlan::TemporalScan {
                start,
                end,
                filters,
                limit,
                order_by,
            } => {
                assert_eq!((start, end), (100, 900), "tightest bounds");
                assert_eq!(filters.len(), 4, "range filters kept for the re-check");
                assert!(filters.iter().any(|f| f.field == Field::Type));
                assert_eq!(limit, Some(10));
                assert!(order_by.is_some_and(|ob| ob.descending));
            }
            other => panic!("expected TemporalScan, got {:?}", other),
        }
        match plan_mql("RECALL memories WHERE created > 5") {
            QueryPlan::TemporalScan { limit, .. } => assert_eq!(limit, Some(DEFAULT_LIMIT)),
            other => panic!("expected TemporalScan, got {:?}", other),
        }
    }

    #[test]
    fn test_accessed_range_is_not_a_creation_index_scan() {
        match plan_mql("RECALL memories WHERE accessed > 100 LIMIT 5") {
            QueryPlan::TagScan { filters, .. } => assert_eq!(filters.len(), 1),
            other => panic!("expected TagScan, got {:?}", other),
        }
    }

    #[test]
    fn test_owner_and_score_filters_are_validated() {
        let id = "550e8400-e29b-41d4-a716-446655440000";
        for ok in [
            format!("RECALL memories WHERE agent = {id}"),
            format!("RECALL memories WHERE agent = \"{id}\""),
            format!("RECALL memories WHERE space != {id}"),
            format!("RECALL memories WHERE agent IN [{id}, {id}]"),
            "RECALL memories WHERE salience > 0.5".to_string(),
            "RECALL memories WHERE confidence <= 1".to_string(),
            "RECALL memories WHERE salience IN [0.1, 0.2]".to_string(),
        ] {
            let tokens = tokenize(&ok).unwrap();
            let stmt = Parser::parse(&tokens).unwrap();
            assert!(plan(&stmt).is_ok(), "{ok}");
        }
        for bad in [
            "RECALL memories WHERE agent = \"alice\"".to_string(),
            format!("RECALL memories WHERE agent < {id}"),
            "RECALL memories WHERE space = 1".to_string(),
            "RECALL memories WHERE salience = \"high\"".to_string(),
            "RECALL memories WHERE confidence CONTAINS 1".to_string(),
            "RECALL memories WHERE confidence IN [1, \"x\"]".to_string(),
            "RECALL memories WHERE NOT (type = episodic OR agent = \"bob\")".to_string(),
            "CONSOLIDATE WHERE salience < \"low\"".to_string(),
        ] {
            let tokens = tokenize(&bad).unwrap();
            let stmt = Parser::parse(&tokens).unwrap();
            assert!(
                matches!(plan(&stmt), Err(MenteError::Query(_))),
                "{bad}: expected a query error"
            );
        }
    }

    #[test]
    fn test_forget_produces_delete() {
        let qp = plan_mql("FORGET 550e8400-e29b-41d4-a716-446655440000");
        match qp {
            QueryPlan::Delete { id } => {
                assert_eq!(
                    id,
                    "550e8400-e29b-41d4-a716-446655440000"
                        .parse::<MemoryId>()
                        .unwrap()
                );
            }
            _ => panic!("expected Delete, got {:?}", qp),
        }
    }

    #[test]
    fn test_traverse_produces_graph_traversal() {
        let qp = plan_mql(
            "TRAVERSE 550e8400-e29b-41d4-a716-446655440000 DEPTH 3 WHERE edge_type = caused",
        );
        match qp {
            QueryPlan::GraphTraversal {
                depth, edge_types, ..
            } => {
                assert_eq!(depth, 3);
                assert_eq!(edge_types, vec![EdgeType::Caused]);
            }
            _ => panic!("expected GraphTraversal, got {:?}", qp),
        }
    }

    #[test]
    fn test_relate_produces_edge_insert() {
        let qp = plan_mql(
            "RELATE 550e8400-e29b-41d4-a716-446655440000 -> 660e8400-e29b-41d4-a716-446655440000 AS caused WITH weight = 0.8",
        );
        match qp {
            QueryPlan::EdgeInsert {
                edge_type, weight, ..
            } => {
                assert_eq!(edge_type, EdgeType::Caused);
                assert!((weight - 0.8).abs() < f32::EPSILON);
            }
            _ => panic!("expected EdgeInsert, got {:?}", qp),
        }
    }
}
