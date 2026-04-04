//! Public MQL entry point: parse an MQL string into a `QueryPlan`.

use mentedb_core::error::MenteResult;

use crate::lexer;
use crate::parser::Parser;
use crate::planner::{self, QueryPlan};

/// The MQL query interface.
///
/// Combines lexing, parsing, and planning into a single call.
#[derive(Debug, Default)]
pub struct Mql;

impl Mql {
    /// Parse an MQL query string and produce an execution plan.
    pub fn parse(input: &str) -> MenteResult<QueryPlan> {
        let tokens = lexer::tokenize(input)?;
        let statement = Parser::parse(&tokens)?;
        planner::plan(&statement)
    }
}
