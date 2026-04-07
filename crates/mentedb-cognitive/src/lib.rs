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
//!   based on conversation trajectory. Uses cosine similarity on embeddings with
//!   keyword overlap as fallback.
//!
//! - [`stream`]: Monitors the LLM's output token stream in real time, comparing
//!   against stored facts to detect contradictions, forgotten knowledge, corrections,
//!   and reinforcements mid-generation.
//!
//! - [`trajectory`]: Tracks the reasoning arc of a conversation as a sequence of
//!   decision states. Learns topic transition patterns via a Markov chain frequency
//!   map that improves predictions over time. Supports resume context generation,
//!   next-topic prediction, and feedback reinforcement from cache hits.
//!
//! - [`write_inference`]: Runs inference at write time to detect contradictions,
//!   create relationship edges, mark obsolete memories, adjust confidence, and
//!   trigger belief propagation automatically.

/// Interference detection between confusable memories.
pub mod interference;
/// Pain signal registry for negative experience tracking.
pub mod pain;
/// Phantom memory detection for knowledge gaps.
pub mod phantom;
/// Speculative context pre assembly cache.
pub mod speculative;
/// Real time LLM output stream monitoring.
pub mod stream;
/// Conversation trajectory tracking and prediction.
pub mod trajectory;
/// Write time inference engine for automatic relationship discovery.
pub mod write_inference;
/// LLM powered cognitive judgment for memory operations.
pub mod llm;

pub use interference::{InterferenceDetector, InterferencePair};
pub use llm::{
    ClusterMember, CognitiveLlmService, ConsolidationDecision, ContradictionVerdict,
    EntityCandidate, EntityMergeGroup, InvalidationVerdict, LlmJudge, LlmJudgeError,
    MemorySummary, MockLlmJudge, TopicLabel,
};
pub use pain::{PainRegistry, PainSignal};
pub use phantom::{EntityRegistry, PhantomConfig, PhantomMemory, PhantomPriority, PhantomTracker};
pub use speculative::{CacheEntry, CacheStats, SpeculativeCache};
pub use stream::{CognitionStream, StreamAlert, StreamConfig, TokenEvent};
pub use trajectory::{DecisionState, TrajectoryNode, TrajectoryTracker, TransitionMap};
pub use write_inference::{InferredAction, WriteInferenceConfig, WriteInferenceEngine};
