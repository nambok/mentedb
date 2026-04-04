//! MenteDB Query — MQL parser and query planner.
//!
//! This crate provides:
//! - MQL (Mente Query Language) lexer and parser
//! - Query planner that produces a `QueryPlan` AST
//! - No direct execution — downstream crates execute plans

pub mod ast;
pub mod lexer;
pub mod mql;
pub mod parser;
pub mod planner;

pub use ast::*;
pub use lexer::{Token, TokenKind};
pub use mql::Mql;
pub use planner::QueryPlan;
