//! AST types for MQL statements.

use mentedb_core::edge::EdgeType;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::MemoryId;
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
    /// The top-level ANDed leaf filters. For a pure `a AND b AND c` WHERE clause
    /// this holds every filter and `condition` is `None`, so the planner keeps all
    /// its index optimizations. When the WHERE uses OR, NOT, or parentheses,
    /// `condition` carries the full boolean tree and this is empty.
    pub filters: Vec<Filter>,
    /// The full boolean WHERE expression, set only when it is not a pure AND of
    /// leaves (i.e. it contains OR, NOT, or grouping). When present, the executor
    /// evaluates this tree; the planner falls back to a full scan and lets the
    /// post-filter decide, so results are always correct even if less optimized.
    pub condition: Option<Condition>,
    pub near: Option<Vec<f32>>,
    pub limit: Option<usize>,
    pub order_by: Option<OrderBy>,
}

/// A boolean combination of filters. `Leaf` is a single comparison; `And`/`Or`
/// combine sub-conditions; `Not` negates one. This is the general form of a WHERE
/// clause; a pure AND of leaves is represented directly as `RecallStatement.filters`
/// instead, so the common case needs no tree walk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Condition {
    And(Vec<Condition>),
    Or(Vec<Condition>),
    Not(Box<Condition>),
    Leaf(Filter),
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
    /// Temporal validity at a point in time: `AS OF <t>` keeps only memories that
    /// were valid at timestamp `t` (valid_from <= t < valid_until), so a query can
    /// see what was true at a past moment, including facts later superseded.
    ValidAt,
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
    /// Set membership: the field equals any element of a list value.
    In,
    /// Substring (content) or membership (tag) test against a text value.
    Contains,
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
    /// A list of values, for the `IN` operator.
    List(Vec<Value>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderBy {
    pub field: Field,
    pub descending: bool,
}
