//! Rendezvous (highest-random-weight) placement.
//!
//! Maps a key to exactly one node from the live set. Unlike `hash(key) % N`,
//! adding or removing a node re-homes only the keys that node owned (about 1/N of
//! the population), not everyone, so the fleet can scale without reshuffling the
//! whole keyspace.

/// FNV-1a 64. A fixed, stable hash so every node in the fleet computes the same
/// owner for a key; a randomized hasher (like the stdlib default) would place keys
/// differently per process and break agreement.
fn weight(key: &str, node: &str) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    for &b in node.as_bytes() {
        hash = (hash ^ b as u64).wrapping_mul(PRIME);
    }
    // Separator so ("ab", "c") and ("a", "bc") do not collide.
    hash = (hash ^ 0x1f).wrapping_mul(PRIME);
    for &b in key.as_bytes() {
        hash = (hash ^ b as u64).wrapping_mul(PRIME);
    }
    hash
}

/// The node that owns `key`, or `None` if the node set is empty. Ties on equal
/// weight break on node id, so the result is fully deterministic.
pub fn owner<'a>(key: &str, nodes: &'a [String]) -> Option<&'a str> {
    nodes
        .iter()
        .map(|n| (weight(key, n), n))
        .max_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)))
        .map(|(_, n)| n.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nodes(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("node-{i}")).collect()
    }

    #[test]
    fn empty_set_has_no_owner() {
        assert_eq!(owner("k1", &[]), None);
    }

    #[test]
    fn single_node_owns_everyone() {
        let ns = nodes(1);
        for k in ["a", "b", "c"] {
            assert_eq!(owner(k, &ns), Some("node-0"));
        }
    }

    #[test]
    fn deterministic() {
        let ns = nodes(5);
        assert_eq!(owner("alice", &ns), owner("alice", &ns));
    }

    #[test]
    fn removing_a_non_owner_never_moves_a_key() {
        let full = nodes(6);
        for i in 0..500 {
            let k = format!("key-{i}");
            let owns = owner(&k, &full).unwrap().to_string();
            for drop in &full {
                if *drop == owns {
                    continue;
                }
                let reduced: Vec<String> = full.iter().filter(|n| *n != drop).cloned().collect();
                assert_eq!(
                    owner(&k, &reduced).unwrap(),
                    owns,
                    "key {k} moved when non-owner {drop} was removed"
                );
            }
        }
    }

    #[test]
    fn removing_a_node_rehomes_only_its_own_keys() {
        let full = nodes(4);
        let reduced: Vec<String> = full.iter().filter(|n| *n != "node-0").cloned().collect();
        let total = 4000;
        let mut moved = 0;
        for i in 0..total {
            let k = format!("key-{i}");
            if owner(&k, &full) != owner(&k, &reduced) {
                moved += 1;
            }
        }
        assert!(
            moved < total / 2,
            "removing one of four nodes moved {moved}/{total} keys, expected ~{}",
            total / 4
        );
    }

    #[test]
    fn distribution_is_roughly_balanced() {
        let ns = nodes(4);
        let mut counts = [0u32; 4];
        for i in 0..8000 {
            let k = format!("key-{i}");
            let o = owner(&k, &ns).unwrap();
            let idx: usize = o.strip_prefix("node-").unwrap().parse().unwrap();
            counts[idx] += 1;
        }
        for c in counts {
            assert!(c > 1200 && c < 2800, "unbalanced: {counts:?}");
        }
    }
}
