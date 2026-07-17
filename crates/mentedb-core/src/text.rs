//! Small string helpers shared across the workspace.

/// Truncate `s` to at most `max_bytes`, snapping the cut down to the nearest
/// UTF-8 character boundary so a multi-byte character is never split.
///
/// Byte-index slicing like `&s[..s.len().min(n)]` panics when byte `n` lands in
/// the middle of a multi-byte character (Cyrillic, CJK, emoji, accents). This
/// returns a valid `&str` in every case, at the cost of up to three fewer bytes.
pub fn truncate_on_char_boundary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_truncates_exactly() {
        assert_eq!(truncate_on_char_boundary("hello world", 5), "hello");
    }

    #[test]
    fn short_strings_pass_through() {
        assert_eq!(truncate_on_char_boundary("hi", 100), "hi");
        assert_eq!(truncate_on_char_boundary("", 5), "");
    }

    #[test]
    fn never_splits_a_multibyte_char() {
        // Each Cyrillic char is 2 bytes; boundaries fall on even byte offsets.
        // Naive `&s[..5]` would panic; we snap down to 4.
        let s = "Привет";
        let out = truncate_on_char_boundary(s, 5);
        assert_eq!(out, "Пр");
        assert!(s.starts_with(out));
    }

    #[test]
    fn handles_emoji_boundaries() {
        // 'a'(1) + emoji(4) + 'b'(1) = 6 bytes; boundaries at 0,1,5,6.
        let s = "a😀b";
        assert_eq!(truncate_on_char_boundary(s, 1), "a");
        assert_eq!(truncate_on_char_boundary(s, 3), "a"); // mid-emoji -> snap to 1
        assert_eq!(truncate_on_char_boundary(s, 5), "a😀");
        assert_eq!(truncate_on_char_boundary(s, 6), "a😀b");
    }

    #[test]
    fn cjk_at_a_splitting_offset_does_not_panic() {
        // Each CJK char is 3 bytes; a 500-byte cut lands mid-char two times in
        // three. This is the exact prod case behind the enrichment fix.
        let s = "记忆".repeat(400); // 2400 bytes, boundaries every 3
        let out = truncate_on_char_boundary(&s, 500);
        assert!(out.len() <= 500);
        assert!(s.starts_with(out));
        assert!(s.is_char_boundary(out.len()));
    }
}
