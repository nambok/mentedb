//! MenteDB Context — attention-aware context assembly engine.
//!
//! This crate provides:
//! - Token-budget-aware context packing
//! - Attention-pattern-aware ordering (U-curve optimization)
//! - Delta-aware serving (only send what changed)
//! - Token-efficient serialization formats

pub mod budget;
pub mod layout;
pub mod delta;
pub mod serializer;
pub mod assembler;

pub use budget::{TokenBudget, BudgetAllocation, ZoneBudgetConfig};
pub use layout::{ScoredMemory, AttentionZone, ContextBlock, ContextLayout, ZoneThresholds};
pub use delta::{DeltaTracker, DeltaResult};
pub use serializer::{ContextSerializer, CompactFormat, StructuredFormat, DeltaFormat};
pub use assembler::{ContextAssembler, AssemblyConfig, ContextWindow, AssemblyMetadata, OutputFormat};
