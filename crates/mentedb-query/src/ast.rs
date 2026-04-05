//! AST types for MQL statements.

use mentedb_core::edge::EdgeType;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::{MemoryId};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Top-level MQL statement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Statement {
    Recall(RecallStatement),
    Relate(RelateStatement),
    Forget(ForgetStatement),
    Consolidate(ConsolidateStatement),
    Traverse(TraverseStatement),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallStatement {
    pub filters: Vec<Filter>,
    pub near: Option<Vec<f32>>,
    pub limit: Option<usize>,
    pub order_by: Option<OrderBy>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelateStatement {
    pub source: MemoryId,
    pub target: MemoryId,
    pub edge_type: EdgeType,
    pub weight: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForgetStatement {
    pub target: MemoryId,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsolidateStatement {
    pub filters: Vec<Filter>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraverseStatement {
    pub start: MemoryId,
    pub depth: usize,
    pub edge_filter: Option<Vec<EdgeType>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Filter {
    pub field: Field,
    pub op: Operator,
    pub value: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Field {
    Content,
    Type,
    Tag,
    Agent,
    Space,
    Salience,
    Confidence,
    Created,
    Accessed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Operator {
    Eq,
    Neq,
    Gt,
    Lt,
    Gte,
    Lte,
    SimilarTo,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Text(String),
    Number(f64),
    Integer(i64),
    Bool(bool),
    Uuid(Uuid),
    Vector(Vec<f32>),
    MemoryType(MemoryType),
    EdgeType(EdgeType),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderBy {
    pub field: Field,
    pub descending: bool,
}
