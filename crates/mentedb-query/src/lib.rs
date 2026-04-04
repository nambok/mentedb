//! MenteDB Query: MQL parser and query planner.
//!
//! This crate provides:
//! - MQL (Mente Query Language) lexer and parser
//! - Query planner that produces a `QueryPlan` AST
//! - No direct execution, downstream crates execute plans

/// Abstract syntax tree types for MQL statements.
pub mod ast;
/// Tokenizer for the MQL language.
pub mod lexer;
/// Top level MQL entry point (parse, plan).
pub mod mql;
/// Recursive descent parser that produces AST nodes.
pub mod parser;
/// Query planner that converts AST into executable plans.
pub mod planner;

pub use ast::*;
pub use lexer::{Token, TokenKind};
pub use mql::Mql;
pub use planner::QueryPlan;
