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
    },
    TagScan {
        tags: Vec<String>,
        filters: Vec<Filter>,
        limit: Option<usize>,
    },
    TemporalScan {
        start: Timestamp,
        end: Timestamp,
        filters: Vec<Filter>,
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
        });
    }

    // Check for SimilarTo operator in filters — implies vector search via text embedding
    if let Some(sim_filter) = recall.filters.iter().find(|f| f.op == Operator::SimilarTo) {
        if let Value::Text(ref _text) = sim_filter.value {
            // The text will need to be embedded at execution time; we still emit VectorSearch
            // with an empty query vec — the executor is responsible for embedding the text.
            let remaining: Vec<Filter> = recall
                .filters
                .iter()
                .filter(|f| f.op != Operator::SimilarTo)
                .cloned()
                .collect();
            return Ok(QueryPlan::VectorSearch {
                query: Vec::new(), // placeholder — executor embeds text
                k: limit,
                filters: remaining,
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
        });
    }

    // If time-range filters exist (created or accessed with range ops), use TemporalScan
    let time_filters: Vec<&Filter> = recall
        .filters
        .iter()
        .filter(|f| {
            (f.field == Field::Created || f.field == Field::Accessed)
                && matches!(
                    f.op,
                    Operator::Gt | Operator::Lt | Operator::Gte | Operator::Lte
                )
        })
        .collect();

    if !time_filters.is_empty() {
        let remaining: Vec<Filter> = recall
            .filters
            .iter()
            .filter(|f| {
                !((f.field == Field::Created || f.field == Field::Accessed)
                    && matches!(
                        f.op,
                        Operator::Gt | Operator::Lt | Operator::Gte | Operator::Lte
                    ))
            })
            .cloned()
            .collect();

        let mut start: Timestamp = 0;
        let mut end: Timestamp = u64::MAX;
        for f in &time_filters {
            if let Value::Text(ref s) = f.value {
                // Simple heuristic: treat text timestamps as orderable strings for now.
                // A real implementation would parse dates. We use 0/MAX as placeholders.
                let _ = s; // acknowledged
            }
            if let Value::Integer(ts) = f.value {
                let ts = ts as u64;
                match f.op {
                    Operator::Gt | Operator::Gte => start = ts,
                    Operator::Lt | Operator::Lte => end = ts,
                    _ => {}
                }
            }
        }

        return Ok(QueryPlan::TemporalScan {
            start,
            end,
            filters: remaining,
        });
    }

    // Fallback: tag scan with no tags (full scan with filters)
    Ok(QueryPlan::TagScan {
        tags: Vec::new(),
        filters: recall.filters.clone(),
        limit: Some(limit),
    })
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
