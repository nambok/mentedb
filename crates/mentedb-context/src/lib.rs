//! MenteDB Context: attention aware context assembly engine.
//!
//! This crate provides:
//! - Token-budget-aware context packing
//! - Attention-pattern-aware ordering (U-curve optimization)
//! - Delta-aware serving (only send what changed)
//! - Token-efficient serialization formats

/// Core context assembly logic.
pub mod assembler;
/// Token budget tracking and allocation.
pub mod budget;
/// Delta aware serving (only send what changed).
pub mod delta;
/// Attention zone layout and scored memory ranking.
pub mod layout;
/// Serialization formats for context output.
pub mod serializer;

pub use assembler::{
    AssemblyConfig, AssemblyMetadata, ContextAssembler, ContextWindow, OutputFormat,
};
pub use budget::{BudgetAllocation, TokenBudget, ZoneBudgetConfig};
pub use delta::{DeltaResult, DeltaTracker};
pub use layout::{AttentionZone, ContextBlock, ContextLayout, ScoredMemory, ZoneThresholds};
pub use serializer::{CompactFormat, ContextSerializer, DeltaFormat, StructuredFormat};
