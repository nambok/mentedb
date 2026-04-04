pub mod interference;
pub mod pain;
pub mod phantom;
pub mod speculative;
pub mod stream;
pub mod trajectory;
pub mod write_inference;

pub use interference::{InterferenceDetector, InterferencePair};
pub use pain::{PainRegistry, PainSignal};
pub use phantom::{PhantomMemory, PhantomPriority, PhantomTracker};
pub use speculative::{CacheEntry, CacheStats, SpeculativeCache};
pub use stream::{CognitionStream, StreamAlert, TokenEvent};
pub use trajectory::{DecisionState, TrajectoryNode, TrajectoryTracker};
pub use write_inference::{InferredAction, WriteInferenceEngine};
