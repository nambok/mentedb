//! MenteDB Core: fundamental types for the mind database.
//!
//! This crate defines the shared data model used by all MenteDB subsystems:
//!
//! - [`MemoryNode`]: The atomic unit of knowledge with embeddings, metadata, and tags
//! - [`MemoryEdge`]: Typed, weighted relationships between memories
//! - [`MemoryTier`]: Cognitive inspired storage hierarchy
//! - [`MenteError`]: Unified error type for the workspace
//! - [`MenteConfig`]: Top level configuration
//! - [`Agent`] and [`MemorySpace`]: Multi tenant isolation primitives
//! - [`EventBus`]: Publish/subscribe event system
//! - [`Version`] and [`VersionStore`]: MVCC versioning for conflict detection

pub mod agent;
pub mod config;
pub mod conflict;
pub mod edge;
pub mod error;
pub mod event;
pub mod limits;
pub mod memory;
pub mod metrics;
pub mod mvcc;
pub mod space;
pub mod tier;
pub mod types;

pub use agent::{Agent, AgentRegistry};
pub use config::MenteConfig;
pub use conflict::{Conflict, ConflictResolver, ConflictVersion, Resolution};
pub use edge::MemoryEdge;
pub use error::MenteError;
pub use event::{EventBus, MenteEvent};
pub use limits::{ResourceLimits, ResourceTracker};
pub use memory::MemoryNode;
pub use metrics::Metrics;
pub use mvcc::{Version, VersionStore};
pub use space::{MemorySpace, Permission, SpaceManager};
pub use tier::MemoryTier;
