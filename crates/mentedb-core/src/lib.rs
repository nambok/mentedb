//! MenteDB Core — fundamental types for the mind database.

pub mod agent;
pub mod conflict;
pub mod edge;
pub mod error;
pub mod event;
pub mod memory;
pub mod mvcc;
pub mod space;
pub mod tier;
pub mod types;

pub use agent::{Agent, AgentRegistry};
pub use conflict::{Conflict, ConflictResolver, ConflictVersion, Resolution};
pub use edge::MemoryEdge;
pub use error::MenteError;
pub use event::{EventBus, MenteEvent};
pub use memory::MemoryNode;
pub use mvcc::{Version, VersionStore};
pub use space::{MemorySpace, Permission, SpaceManager};
pub use tier::MemoryTier;
