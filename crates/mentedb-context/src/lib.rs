//! MenteDB Context — attention-aware context assembly engine.
//!
//! This crate provides:
//! - Token-budget-aware context packing
//! - Attention-pattern-aware ordering (U-curve optimization)
//! - Delta-aware serving (only send what changed)
//! - Token-efficient serialization formats
