//! Hand-written lexer for MQL.

use mentedb_core::error::{MenteError, MenteResult};

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub lexeme: String,
    pub position: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenKind {
    // Statements
    Recall,
    Relate,
    Forget,
    Consolidate,
    Traverse,

    // Clauses
    Where,
    And,
    Or,
    Not,
    Near,
    Within,
    Limit,
    OrderBy,
    As,
    From,
    To,
    With,

    // Keywords
    Agent,
    Space,
    Type,
    Tag,
    Salience,
    Confidence,
    Created,
    Accessed,
    Depth,
    Hops,
    Memories,
    By,
    EdgeType,

    // Operators
    Eq,       // =
    Neq,      // !=
    Gt,       // >
    Lt,       // <
    Gte,      // >=
    Lte,      // <=
    SimilarTo, // ~>
    Arrow,    // ->

    // Punctuation
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Dot,
    Colon,
    Semicolon,

    // Literals
    StringLit,
    IntegerLit,
    FloatLit,
    Identifier,
    UuidLit,

    Eof,
}

pub fn tokenize(input: &str) -> MenteResult<Vec<Token>> {
    let mut tokens = Vec::new();
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut pos = 0;

    while pos < len {
        // Skip whitespace
        if bytes[pos].is_ascii_whitespace() {
            pos += 1;
            continue;
        }

        let start = pos;

        // String literal
        if bytes[pos] == b'"' {
            pos += 1;
            while pos < len && bytes[pos] != b'"' {
                if bytes[pos] == b'\\' {
                    pos += 1; // skip escaped char
                }
                pos += 1;
            }
            if pos >= len {
                return Err(MenteError::Query("unterminated string literal".into()));
            }
            pos += 1; // closing quote
            let lexeme = input[start..pos].to_string();
            tokens.push(Token { kind: TokenKind::StringLit, lexeme, position: start });
            continue;
        }

        // Two-char operators
        if pos + 1 < len {
            let two = &input[start..start + 2];
            let kind = match two {
                "!=" => Some(TokenKind::Neq),
                ">=" => Some(TokenKind::Gte),
                "<=" => Some(TokenKind::Lte),
                "~>" => Some(TokenKind::SimilarTo),
                "->" => Some(TokenKind::Arrow),
                _ => None,
            };
            if let Some(k) = kind {
                tokens.push(Token { kind: k, lexeme: two.to_string(), position: start });
                pos += 2;
                continue;
            }
        }

        // Single-char operators/punctuation
        let single = match bytes[pos] {
            b'=' => Some(TokenKind::Eq),
            b'>' => Some(TokenKind::Gt),
            b'<' => Some(TokenKind::Lt),
            b'(' => Some(TokenKind::LParen),
            b')' => Some(TokenKind::RParen),
            b'[' => Some(TokenKind::LBracket),
            b']' => Some(TokenKind::RBracket),
            b',' => Some(TokenKind::Comma),
            b'.' => Some(TokenKind::Dot),
            b':' => Some(TokenKind::Colon),
            b';' => Some(TokenKind::Semicolon),
            _ => None,
        };
        if let Some(k) = single {
            tokens.push(Token {
                kind: k,
                lexeme: input[start..start + 1].to_string(),
                position: start,
            });
            pos += 1;
            continue;
        }

        // Try UUID first: if we see a hex digit, speculatively scan for UUID pattern
        if bytes[pos].is_ascii_hexdigit() {
            let saved = pos;
            // Consume alphanumeric + hyphens to check for UUID
            while pos < len
                && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_' || bytes[pos] == b'-')
            {
                pos += 1;
            }
            let candidate = &input[saved..pos];
            if is_uuid_like(candidate) {
                tokens.push(Token {
                    kind: TokenKind::UuidLit,
                    lexeme: candidate.to_string(),
                    position: start,
                });
                continue;
            }
            // Not a UUID — reset and fall through to number/identifier parsing
            pos = saved;
        }

        // Numbers (may start with - for negative)
        if bytes[pos].is_ascii_digit()
            || (bytes[pos] == b'-' && pos + 1 < len && bytes[pos + 1].is_ascii_digit())
        {
            if bytes[pos] == b'-' {
                pos += 1;
            }
            while pos < len && bytes[pos].is_ascii_digit() {
                pos += 1;
            }
            let mut is_float = false;
            if pos < len && bytes[pos] == b'.' && pos + 1 < len && bytes[pos + 1].is_ascii_digit()
            {
                is_float = true;
                pos += 1;
                while pos < len && bytes[pos].is_ascii_digit() {
                    pos += 1;
                }
            }
            let lexeme = input[start..pos].to_string();
            let kind = if is_float { TokenKind::FloatLit } else { TokenKind::IntegerLit };
            tokens.push(Token { kind, lexeme, position: start });
            continue;
        }

        // Identifiers, keywords
        if bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_' {
            while pos < len && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_' || bytes[pos] == b'-') {
                pos += 1;
            }
            let lexeme = input[start..pos].to_string();

            let kind = match lexeme.to_lowercase().as_str() {
                "recall" => TokenKind::Recall,
                "relate" => TokenKind::Relate,
                "forget" => TokenKind::Forget,
                "consolidate" => TokenKind::Consolidate,
                "traverse" => TokenKind::Traverse,
                "where" => TokenKind::Where,
                "and" => TokenKind::And,
                "or" => TokenKind::Or,
                "not" => TokenKind::Not,
                "near" => TokenKind::Near,
                "within" => TokenKind::Within,
                "limit" => TokenKind::Limit,
                "order" => TokenKind::OrderBy,
                "as" => TokenKind::As,
                "from" => TokenKind::From,
                "to" => TokenKind::To,
                "with" => TokenKind::With,
                "agent" => TokenKind::Agent,
                "space" => TokenKind::Space,
                "type" => TokenKind::Type,
                "tag" => TokenKind::Tag,
                "salience" => TokenKind::Salience,
                "confidence" => TokenKind::Confidence,
                "created" => TokenKind::Created,
                "accessed" => TokenKind::Accessed,
                "depth" => TokenKind::Depth,
                "hops" => TokenKind::Hops,
                "memories" => TokenKind::Memories,
                "by" => TokenKind::By,
                "edge_type" => TokenKind::EdgeType,
                _ => TokenKind::Identifier,
            };
            tokens.push(Token { kind, lexeme, position: start });
            continue;
        }

        return Err(MenteError::Query(format!(
            "unexpected character '{}' at position {}",
            bytes[pos] as char, pos
        )));
    }

    tokens.push(Token { kind: TokenKind::Eof, lexeme: String::new(), position: pos });
    Ok(tokens)
}

fn is_uuid_like(s: &str) -> bool {
    // UUID format: 8-4-4-4-12 hex chars (with dashes)
    if s.len() != 36 {
        return false;
    }
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 5 {
        return false;
    }
    let expected_lens = [8, 4, 4, 4, 12];
    for (part, &expected) in parts.iter().zip(&expected_lens) {
        if part.len() != expected || !part.chars().all(|c| c.is_ascii_hexdigit()) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_statement_tokens() {
        let tokens = tokenize("RECALL memories WHERE type = episodic LIMIT 10").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Recall);
        assert_eq!(tokens[1].kind, TokenKind::Memories);
        assert_eq!(tokens[2].kind, TokenKind::Where);
        assert_eq!(tokens[3].kind, TokenKind::Type);
        assert_eq!(tokens[4].kind, TokenKind::Eq);
        assert_eq!(tokens[5].kind, TokenKind::Identifier);
        assert_eq!(tokens[5].lexeme, "episodic");
        assert_eq!(tokens[6].kind, TokenKind::Limit);
        assert_eq!(tokens[7].kind, TokenKind::IntegerLit);
        assert_eq!(tokens[8].kind, TokenKind::Eof);
    }

    #[test]
    fn test_string_literal() {
        let tokens = tokenize(r#"content ~> "database migration""#).unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Identifier);
        assert_eq!(tokens[1].kind, TokenKind::SimilarTo);
        assert_eq!(tokens[2].kind, TokenKind::StringLit);
        assert_eq!(tokens[2].lexeme, r#""database migration""#);
    }

    #[test]
    fn test_operators() {
        let tokens = tokenize("= != > < >= <= ~> ->").unwrap();
        let kinds: Vec<TokenKind> = tokens.iter().map(|t| t.kind).collect();
        assert_eq!(kinds, vec![
            TokenKind::Eq, TokenKind::Neq, TokenKind::Gt, TokenKind::Lt,
            TokenKind::Gte, TokenKind::Lte, TokenKind::SimilarTo, TokenKind::Arrow,
            TokenKind::Eof,
        ]);
    }

    #[test]
    fn test_uuid_token() {
        let tokens = tokenize("550e8400-e29b-41d4-a716-446655440000").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::UuidLit);
    }

    #[test]
    fn test_float_literal() {
        let tokens = tokenize("0.1 42 3.14").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::FloatLit);
        assert_eq!(tokens[1].kind, TokenKind::IntegerLit);
        assert_eq!(tokens[2].kind, TokenKind::FloatLit);
    }

    #[test]
    fn test_vector_literal() {
        let tokens = tokenize("[0.1, 0.2, 0.3]").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::LBracket);
        assert_eq!(tokens[1].kind, TokenKind::FloatLit);
        assert_eq!(tokens[2].kind, TokenKind::Comma);
        assert_eq!(tokens[5].kind, TokenKind::FloatLit);
        assert_eq!(tokens[6].kind, TokenKind::RBracket);
    }

    #[test]
    fn test_punctuation() {
        let tokens = tokenize("( ) [ ] , . : ;").unwrap();
        let kinds: Vec<TokenKind> = tokens.iter().map(|t| t.kind).collect();
        assert_eq!(kinds, vec![
            TokenKind::LParen, TokenKind::RParen, TokenKind::LBracket, TokenKind::RBracket,
            TokenKind::Comma, TokenKind::Dot, TokenKind::Colon, TokenKind::Semicolon,
            TokenKind::Eof,
        ]);
    }
}
