//! End-to-end integration tests for MQL parsing.

use mentedb_core::edge::EdgeType;
use mentedb_core::memory::MemoryType;
use mentedb_query::Mql;
use mentedb_query::ast::*;
use mentedb_query::planner::QueryPlan;

#[test]
fn test_recall_with_similar_to_produces_vector_search() {
    let plan = Mql::parse(
        r#"RECALL memories WHERE content ~> "database migration" AND tag = "backend" LIMIT 10"#,
    )
    .unwrap();
    match plan {
        QueryPlan::VectorSearch { k, filters, .. } => {
            assert_eq!(k, 10);
            // The tag filter should remain in filters
            assert_eq!(filters.len(), 1);
            assert_eq!(filters[0].field, Field::Tag);
        }
        other => panic!("expected VectorSearch, got {:?}", other),
    }
}

#[test]
fn test_recall_with_type_and_order_by() {
    let plan =
        Mql::parse("RECALL memories WHERE type = episodic ORDER BY salience LIMIT 5").unwrap();
    // type = episodic without tags or vector → fallback TagScan
    match plan {
        QueryPlan::TagScan { filters, limit, .. } => {
            assert_eq!(limit, Some(5));
            assert_eq!(filters.len(), 1);
            assert_eq!(filters[0].value, Value::MemoryType(MemoryType::Episodic));
        }
        other => panic!("expected TagScan, got {:?}", other),
    }
}

#[test]
fn test_recall_near_vector() {
    let plan = Mql::parse(
        r#"RECALL memories NEAR [0.1, 0.2, 0.3] WHERE agent = "550e8400-e29b-41d4-a716-446655440000" LIMIT 10"#,
    )
    .unwrap();
    match plan {
        QueryPlan::VectorSearch { query, k, filters } => {
            assert_eq!(query, vec![0.1, 0.2, 0.3]);
            assert_eq!(k, 10);
            assert_eq!(filters.len(), 1);
            assert_eq!(filters[0].field, Field::Agent);
        }
        other => panic!("expected VectorSearch, got {:?}", other),
    }
}

#[test]
fn test_relate_end_to_end() {
    let plan = Mql::parse(
        "RELATE 550e8400-e29b-41d4-a716-446655440000 -> 660e8400-e29b-41d4-a716-446655440000 AS caused WITH weight = 0.9",
    )
    .unwrap();
    match plan {
        QueryPlan::EdgeInsert {
            edge_type, weight, ..
        } => {
            assert_eq!(edge_type, EdgeType::Caused);
            assert!((weight - 0.9).abs() < f32::EPSILON);
        }
        other => panic!("expected EdgeInsert, got {:?}", other),
    }
}

#[test]
fn test_forget_end_to_end() {
    let plan = Mql::parse("FORGET 550e8400-e29b-41d4-a716-446655440000").unwrap();
    assert!(matches!(plan, QueryPlan::Delete { .. }));
}

#[test]
fn test_consolidate_end_to_end() {
    let plan =
        Mql::parse(r#"CONSOLIDATE WHERE type = episodic AND accessed < "2024-01-01""#).unwrap();
    match plan {
        QueryPlan::Consolidate { filters } => {
            assert_eq!(filters.len(), 2);
        }
        other => panic!("expected Consolidate, got {:?}", other),
    }
}

#[test]
fn test_traverse_end_to_end() {
    let plan = Mql::parse(
        "TRAVERSE 550e8400-e29b-41d4-a716-446655440000 DEPTH 3 WHERE edge_type = caused",
    )
    .unwrap();
    match plan {
        QueryPlan::GraphTraversal {
            depth, edge_types, ..
        } => {
            assert_eq!(depth, 3);
            assert_eq!(edge_types, vec![EdgeType::Caused]);
        }
        other => panic!("expected GraphTraversal, got {:?}", other),
    }
}
