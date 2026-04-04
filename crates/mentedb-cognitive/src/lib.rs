//! # mentedb-cognitive: Cognitive Engine for MenteDB
//!
//! This crate implements the seven cognitive features that distinguish MenteDB
//! from a conventional vector database. These features run at write time and
//! read time, giving the engine awareness of the knowledge it holds.
//!
//! ## Modules
//!
//! - [`interference`]: Detects pairs of memories similar enough to confuse an LLM,
//!   generates disambiguation text, and reorders context to maximize separation.
//!
//! - [`pain`]: Records negative experiences (failed actions, user corrections) as
//!   pain signals with intensity and decay. Surfaces warnings during context assembly.
//!
//! - [`phantom`]: Detects references to entities not present in the knowledge base
//!   (phantom memories). Flags gaps so the agent can acquire missing knowledge.
//!
//! - [`speculative`]: Pre-assembles context windows for predicted upcoming queries
//!   based on conversation trajectory. Uses Jaccard keyword overlap for cache hits.
//!
//! - [`stream`]: Monitors the LLM's output token stream in real time, comparing
//!   against stored facts to detect contradictions, forgotten knowledge, corrections,
//!   and reinforcements mid-generation.
//!
//! - [`trajectory`]: Tracks the reasoning arc of a conversation as a sequence of
//!   decision states. Supports resume context generation and next-topic prediction.
//!
//! - [`write_inference`]: Runs inference at write time to detect contradictions,
//!   create relationship edges, mark obsolete memories, adjust confidence, and
//!   trigger belief propagation automatically.

pub mod interference;
pub mod pain;
pub mod phantom;
pub mod speculative;
pub mod stream;
pub mod trajectory;
pub mod write_inference;

pub use interference::{InterferenceDetector, InterferencePair};
pub use pain::{PainRegistry, PainSignal};
pub use phantom::{EntityRegistry, PhantomConfig, PhantomMemory, PhantomPriority, PhantomTracker};
pub use speculative::{CacheEntry, CacheStats, SpeculativeCache};
pub use stream::{CognitionStream, StreamAlert, StreamConfig, TokenEvent};
pub use trajectory::{DecisionState, TrajectoryNode, TrajectoryTracker};
pub use write_inference::{InferredAction, WriteInferenceConfig, WriteInferenceEngine};
