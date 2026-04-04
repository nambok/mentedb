//! Event System — publish/subscribe bus for memory-graph events.

use parking_lot::RwLock;

use crate::edge::EdgeType;
use crate::types::{AgentId, MemoryId, SpaceId};

/// A unique subscriber handle returned by [`EventBus::subscribe`].
pub type SubscriberId = usize;

/// Events emitted by the memory system.
#[derive(Debug, Clone)]
pub enum MenteEvent {
    MemoryCreated {
        id: MemoryId,
        agent_id: AgentId,
    },
    MemoryUpdated {
        id: MemoryId,
        version: u64,
    },
    MemoryDeleted {
        id: MemoryId,
    },
    EdgeCreated {
        source: MemoryId,
        target: MemoryId,
        edge_type: EdgeType,
    },
    BeliefChanged {
        id: MemoryId,
        old_confidence: f32,
        new_confidence: f32,
    },
    SpaceCreated {
        id: SpaceId,
    },
    ContradictionDetected {
        a: MemoryId,
        b: MemoryId,
    },
}

/// Thread-safe event bus.
type Subscriber = Box<dyn Fn(&MenteEvent) + Send + Sync>;

struct Entry {
    id: SubscriberId,
    callback: Subscriber,
}

/// A publish/subscribe event bus protected by `parking_lot::RwLock`.
pub struct EventBus {
    subscribers: RwLock<Vec<Entry>>,
    next_id: RwLock<usize>,
}

impl std::fmt::Debug for EventBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventBus")
            .field("subscriber_count", &self.subscribers.read().len())
            .finish()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self {
            subscribers: RwLock::new(Vec::new()),
            next_id: RwLock::new(0),
        }
    }
}

impl EventBus {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a subscriber callback. Returns a handle for unsubscription.
    pub fn subscribe(
        &self,
        callback: impl Fn(&MenteEvent) + Send + Sync + 'static,
    ) -> SubscriberId {
        let mut next = self.next_id.write();
        let id = *next;
        *next += 1;
        self.subscribers.write().push(Entry {
            id,
            callback: Box::new(callback),
        });
        id
    }

    /// Remove a subscriber by handle.
    pub fn unsubscribe(&self, id: SubscriberId) {
        self.subscribers.write().retain(|e| e.id != id);
    }

    /// Publish an event to all current subscribers.
    pub fn publish(&self, event: MenteEvent) {
        let subs = self.subscribers.read();
        for entry in subs.iter() {
            (entry.callback)(&event);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use uuid::Uuid;

    #[test]
    fn subscribe_and_publish() {
        let bus = EventBus::new();
        let count = Arc::new(AtomicUsize::new(0));
        let c = count.clone();
        bus.subscribe(move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        bus.publish(MenteEvent::SpaceCreated { id: Uuid::new_v4() });
        assert_eq!(count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn unsubscribe() {
        let bus = EventBus::new();
        let count = Arc::new(AtomicUsize::new(0));
        let c = count.clone();
        let sid = bus.subscribe(move |_| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        bus.unsubscribe(sid);
        bus.publish(MenteEvent::SpaceCreated { id: Uuid::new_v4() });
        assert_eq!(count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn multiple_subscribers() {
        let bus = EventBus::new();
        let count = Arc::new(AtomicUsize::new(0));
        for _ in 0..3 {
            let c = count.clone();
            bus.subscribe(move |_| {
                c.fetch_add(1, Ordering::Relaxed);
            });
        }
        bus.publish(MenteEvent::MemoryDeleted { id: Uuid::new_v4() });
        assert_eq!(count.load(Ordering::Relaxed), 3);
    }
}
