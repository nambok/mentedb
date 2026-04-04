//! Token budget management for context assembly.

/// Estimates token count for a text string.
/// Uses word count * 1.3 as a rough approximation for English text.
fn estimate_tokens(text: &str) -> usize {
    let word_count = text.split_whitespace().count();
    ((word_count as f64) * 1.3).ceil() as usize
}

/// Tracks token usage against a maximum budget.
#[derive(Debug, Clone)]
pub struct TokenBudget {
    pub max_tokens: usize,
    pub used_tokens: usize,
}

impl TokenBudget {
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            used_tokens: 0,
        }
    }

    /// Remaining tokens available.
    pub fn remaining(&self) -> usize {
        self.max_tokens.saturating_sub(self.used_tokens)
    }

    /// Check if the given text fits within remaining budget.
    pub fn can_fit(&self, text: &str) -> bool {
        estimate_tokens(text) <= self.remaining()
    }

    /// Consume tokens for the given text. Returns the number of tokens consumed.
    pub fn consume(&mut self, text: &str) -> usize {
        let tokens = estimate_tokens(text);
        let actual = tokens.min(self.remaining());
        self.used_tokens += actual;
        actual
    }

    /// Reset the budget to zero usage.
    pub fn reset(&mut self) {
        self.used_tokens = 0;
    }
}

/// Divides a total token budget across context zones.
#[derive(Debug, Clone)]
pub struct BudgetAllocation {
    /// System zone: 10%
    pub system: usize,
    /// Critical zone: 25%
    pub critical: usize,
    /// Primary zone: 35%
    pub primary: usize,
    /// Supporting zone: 20%
    pub supporting: usize,
    /// Reference zone: 10%
    pub reference: usize,
}

impl BudgetAllocation {
    /// Allocate a total budget across zones with fixed percentages.
    pub fn from_total(total: usize) -> Self {
        Self {
            system: total / 10,           // 10%
            critical: total / 4,          // 25%
            primary: total * 35 / 100,    // 35%
            supporting: total / 5,        // 20%
            reference: total / 10,        // 10%
        }
    }

    /// Total allocated tokens (may differ slightly from input due to rounding).
    pub fn total(&self) -> usize {
        self.system + self.critical + self.primary + self.supporting + self.reference
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("hello world"), 3); // 2 * 1.3 = 2.6 -> 3
        assert_eq!(estimate_tokens("one"), 2); // 1 * 1.3 = 1.3 -> 2
    }

    #[test]
    fn test_token_budget_lifecycle() {
        let mut budget = TokenBudget::new(100);
        assert_eq!(budget.remaining(), 100);
        assert!(budget.can_fit("hello world"));

        let used = budget.consume("hello world");
        assert_eq!(used, 3);
        assert_eq!(budget.remaining(), 97);

        budget.reset();
        assert_eq!(budget.remaining(), 100);
    }

    #[test]
    fn test_budget_overflow_protection() {
        let mut budget = TokenBudget::new(2);
        // "a b c d e" = 5 words * 1.3 = 7 tokens, won't fit
        assert!(!budget.can_fit("a b c d e"));
        // consume should only take what's available
        let used = budget.consume("a b c d e");
        assert_eq!(used, 2);
        assert_eq!(budget.remaining(), 0);
    }

    #[test]
    fn test_budget_allocation() {
        let alloc = BudgetAllocation::from_total(1000);
        assert_eq!(alloc.system, 100);
        assert_eq!(alloc.critical, 250);
        assert_eq!(alloc.primary, 350);
        assert_eq!(alloc.supporting, 200);
        assert_eq!(alloc.reference, 100);
        assert_eq!(alloc.total(), 1000);
    }
}
