//! Hand-written recursive descent parser for MQL.

use mentedb_core::edge::EdgeType;
use mentedb_core::error::{MenteError, MenteResult};
use mentedb_core::memory::MemoryType;
use uuid::Uuid;

use crate::ast::*;
use crate::lexer::{Token, TokenKind};

pub struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
}

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [Token]) -> Self {
        Self { tokens, pos: 0 }
    }

    pub fn parse(tokens: &[Token]) -> MenteResult<Statement> {
        let mut parser = Parser::new(tokens);
        parser.parse_statement()
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.pos.min(self.tokens.len() - 1)];
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    fn expect(&mut self, kind: TokenKind) -> MenteResult<&Token> {
        let tok = self.peek();
        if tok.kind != kind {
            return Err(MenteError::Query(format!(
                "expected {:?}, found {:?} ('{}') at position {}",
                kind, tok.kind, tok.lexeme, tok.position
            )));
        }
        Ok(self.advance())
    }

    fn at(&self, kind: TokenKind) -> bool {
        self.peek().kind == kind
    }

    fn parse_statement(&mut self) -> MenteResult<Statement> {
        match self.peek().kind {
            TokenKind::Recall => self.parse_recall(),
            TokenKind::Relate => self.parse_relate(),
            TokenKind::Forget => self.parse_forget(),
            TokenKind::Consolidate => self.parse_consolidate(),
            TokenKind::Traverse => self.parse_traverse(),
            _ => Err(MenteError::Query(format!(
                "expected statement keyword, found {:?} at position {}",
                self.peek().kind,
                self.peek().position
            ))),
        }
    }

    fn parse_recall(&mut self) -> MenteResult<Statement> {
        self.advance(); // RECALL

        // Optional "memories" keyword
        if self.at(TokenKind::Memories) {
            self.advance();
        }

        let mut near = None;
        let mut filters = Vec::new();
        let mut limit = None;
        let mut order_by = None;

        // NEAR [vector]
        if self.at(TokenKind::Near) {
            self.advance();
            near = Some(self.parse_vector()?);
        }

        // WHERE clause
        if self.at(TokenKind::Where) {
            self.advance();
            filters = self.parse_filters()?;
        }

        // ORDER BY field
        if self.at(TokenKind::OrderBy) {
            self.advance();
            // consume optional "BY"
            if self.at(TokenKind::By) {
                self.advance();
            }
            let field = self.parse_field()?;
            let descending = false; // default ascending
            order_by = Some(OrderBy { field, descending });
        }

        // LIMIT n
        if self.at(TokenKind::Limit) {
            self.advance();
            let tok = self.advance();
            let n: usize = tok.lexeme.parse().map_err(|_| {
                MenteError::Query(format!("invalid limit value: {}", tok.lexeme))
            })?;
            limit = Some(n);
        }

        Ok(Statement::Recall(RecallStatement { filters, near, limit, order_by }))
    }

    fn parse_relate(&mut self) -> MenteResult<Statement> {
        self.advance(); // RELATE

        let source = self.parse_uuid()?;
        self.expect(TokenKind::Arrow)?;
        let target = self.parse_uuid()?;
        self.expect(TokenKind::As)?;
        let edge_type = self.parse_edge_type()?;

        let mut weight = None;
        if self.at(TokenKind::With) {
            self.advance();
            // expect "weight = <float>"
            self.expect(TokenKind::Identifier)?; // "weight"
            self.expect(TokenKind::Eq)?;
            let tok = self.advance();
            let w: f32 = tok.lexeme.parse().map_err(|_| {
                MenteError::Query(format!("invalid weight value: {}", tok.lexeme))
            })?;
            weight = Some(w);
        }

        Ok(Statement::Relate(RelateStatement { source, target, edge_type, weight }))
    }

    fn parse_forget(&mut self) -> MenteResult<Statement> {
        self.advance(); // FORGET
        let target = self.parse_uuid()?;
        Ok(Statement::Forget(ForgetStatement { target }))
    }

    fn parse_consolidate(&mut self) -> MenteResult<Statement> {
        self.advance(); // CONSOLIDATE
        let mut filters = Vec::new();
        if self.at(TokenKind::Where) {
            self.advance();
            filters = self.parse_filters()?;
        }
        Ok(Statement::Consolidate(ConsolidateStatement { filters }))
    }

    fn parse_traverse(&mut self) -> MenteResult<Statement> {
        self.advance(); // TRAVERSE
        let start = self.parse_uuid()?;

        self.expect(TokenKind::Depth)?;
        let tok = self.advance();
        let depth: usize = tok.lexeme.parse().map_err(|_| {
            MenteError::Query(format!("invalid depth value: {}", tok.lexeme))
        })?;

        let mut edge_filter = None;
        if self.at(TokenKind::Where) {
            self.advance();
            // edge_type = <type>
            self.expect(TokenKind::EdgeType)?;
            self.expect(TokenKind::Eq)?;
            let et = self.parse_edge_type()?;
            edge_filter = Some(vec![et]);
        }

        Ok(Statement::Traverse(TraverseStatement { start, depth, edge_filter }))
    }

    fn parse_filters(&mut self) -> MenteResult<Vec<Filter>> {
        let mut filters = vec![self.parse_filter()?];
        while self.at(TokenKind::And) {
            self.advance();
            filters.push(self.parse_filter()?);
        }
        Ok(filters)
    }

    fn parse_filter(&mut self) -> MenteResult<Filter> {
        let field = self.parse_field()?;
        let op = self.parse_operator()?;
        let value = self.parse_value(&field)?;
        Ok(Filter { field, op, value })
    }

    fn parse_field(&mut self) -> MenteResult<Field> {
        let tok = self.advance();
        match tok.kind {
            TokenKind::Identifier if tok.lexeme.eq_ignore_ascii_case("content") => {
                Ok(Field::Content)
            }
            TokenKind::Type => Ok(Field::Type),
            TokenKind::Tag => Ok(Field::Tag),
            TokenKind::Agent => Ok(Field::Agent),
            TokenKind::Space => Ok(Field::Space),
            TokenKind::Salience => Ok(Field::Salience),
            TokenKind::Confidence => Ok(Field::Confidence),
            TokenKind::Created => Ok(Field::Created),
            TokenKind::Accessed => Ok(Field::Accessed),
            _ => Err(MenteError::Query(format!(
                "expected field name, found '{}' at position {}",
                tok.lexeme, tok.position
            ))),
        }
    }

    fn parse_operator(&mut self) -> MenteResult<Operator> {
        let tok = self.advance();
        match tok.kind {
            TokenKind::Eq => Ok(Operator::Eq),
            TokenKind::Neq => Ok(Operator::Neq),
            TokenKind::Gt => Ok(Operator::Gt),
            TokenKind::Lt => Ok(Operator::Lt),
            TokenKind::Gte => Ok(Operator::Gte),
            TokenKind::Lte => Ok(Operator::Lte),
            TokenKind::SimilarTo => Ok(Operator::SimilarTo),
            _ => Err(MenteError::Query(format!(
                "expected operator, found '{}' at position {}",
                tok.lexeme, tok.position
            ))),
        }
    }

    fn parse_value(&mut self, field: &Field) -> MenteResult<Value> {
        // For Type field, parse as MemoryType
        if *field == Field::Type {
            return self.parse_memory_type_value();
        }

        let tok = self.advance();
        match tok.kind {
            TokenKind::StringLit => {
                // Strip surrounding quotes
                let inner = tok.lexeme[1..tok.lexeme.len() - 1].to_string();
                // Check if this looks like a UUID inside quotes
                if let Ok(uuid) = inner.parse::<Uuid>() {
                    return Ok(Value::Uuid(uuid));
                }
                Ok(Value::Text(inner))
            }
            TokenKind::IntegerLit => {
                let n: i64 = tok.lexeme.parse().map_err(|_| {
                    MenteError::Query(format!("invalid integer: {}", tok.lexeme))
                })?;
                Ok(Value::Integer(n))
            }
            TokenKind::FloatLit => {
                let n: f64 = tok.lexeme.parse().map_err(|_| {
                    MenteError::Query(format!("invalid float: {}", tok.lexeme))
                })?;
                Ok(Value::Number(n))
            }
            TokenKind::UuidLit => {
                let uuid: Uuid = tok.lexeme.parse().map_err(|_| {
                    MenteError::Query(format!("invalid UUID: {}", tok.lexeme))
                })?;
                Ok(Value::Uuid(uuid))
            }
            TokenKind::Identifier => {
                let lower = tok.lexeme.to_lowercase();
                match lower.as_str() {
                    "true" => Ok(Value::Bool(true)),
                    "false" => Ok(Value::Bool(false)),
                    _ => Ok(Value::Text(tok.lexeme.clone())),
                }
            }
            TokenKind::LBracket => {
                // put back and parse as vector
                self.pos -= 1;
                let v = self.parse_vector()?;
                Ok(Value::Vector(v))
            }
            _ => Err(MenteError::Query(format!(
                "expected value, found '{}' at position {}",
                tok.lexeme, tok.position
            ))),
        }
    }

    fn parse_memory_type_value(&mut self) -> MenteResult<Value> {
        let tok = self.advance();
        let name = match tok.kind {
            TokenKind::Identifier | TokenKind::StringLit => {
                let s = if tok.kind == TokenKind::StringLit {
                    tok.lexeme[1..tok.lexeme.len() - 1].to_string()
                } else {
                    tok.lexeme.clone()
                };
                s
            }
            _ => {
                return Err(MenteError::Query(format!(
                    "expected memory type, found '{}' at position {}",
                    tok.lexeme, tok.position
                )));
            }
        };

        let mt = match name.to_lowercase().as_str() {
            "episodic" => MemoryType::Episodic,
            "semantic" => MemoryType::Semantic,
            "procedural" => MemoryType::Procedural,
            "antipattern" | "anti_pattern" => MemoryType::AntiPattern,
            "reasoning" => MemoryType::Reasoning,
            "correction" => MemoryType::Correction,
            _ => {
                return Err(MenteError::Query(format!("unknown memory type: {}", name)));
            }
        };
        Ok(Value::MemoryType(mt))
    }

    fn parse_edge_type(&mut self) -> MenteResult<EdgeType> {
        let tok = self.advance();
        let name = match tok.kind {
            TokenKind::Identifier | TokenKind::StringLit => {
                if tok.kind == TokenKind::StringLit {
                    tok.lexeme[1..tok.lexeme.len() - 1].to_string()
                } else {
                    tok.lexeme.clone()
                }
            }
            _ => {
                return Err(MenteError::Query(format!(
                    "expected edge type, found '{}' at position {}",
                    tok.lexeme, tok.position
                )));
            }
        };

        match name.to_lowercase().as_str() {
            "caused" => Ok(EdgeType::Caused),
            "before" => Ok(EdgeType::Before),
            "related" => Ok(EdgeType::Related),
            "contradicts" => Ok(EdgeType::Contradicts),
            "supports" => Ok(EdgeType::Supports),
            "supersedes" => Ok(EdgeType::Supersedes),
            "derived" => Ok(EdgeType::Derived),
            "partof" | "part_of" => Ok(EdgeType::PartOf),
            _ => Err(MenteError::Query(format!("unknown edge type: {}", name))),
        }
    }

    fn parse_uuid(&mut self) -> MenteResult<Uuid> {
        let tok = self.advance();
        match tok.kind {
            TokenKind::UuidLit => tok.lexeme.parse().map_err(|_| {
                MenteError::Query(format!("invalid UUID: {}", tok.lexeme))
            }),
            TokenKind::StringLit => {
                let inner = &tok.lexeme[1..tok.lexeme.len() - 1];
                inner.parse().map_err(|_| {
                    MenteError::Query(format!("invalid UUID in string: {}", tok.lexeme))
                })
            }
            _ => Err(MenteError::Query(format!(
                "expected UUID, found '{}' at position {}",
                tok.lexeme, tok.position
            ))),
        }
    }

    fn parse_vector(&mut self) -> MenteResult<Vec<f32>> {
        self.expect(TokenKind::LBracket)?;
        let mut values = Vec::new();
        if !self.at(TokenKind::RBracket) {
            let tok = self.advance();
            let v: f32 = tok.lexeme.parse().map_err(|_| {
                MenteError::Query(format!("invalid float in vector: {}", tok.lexeme))
            })?;
            values.push(v);
            while self.at(TokenKind::Comma) {
                self.advance();
                let tok = self.advance();
                let v: f32 = tok.lexeme.parse().map_err(|_| {
                    MenteError::Query(format!("invalid float in vector: {}", tok.lexeme))
                })?;
                values.push(v);
            }
        }
        self.expect(TokenKind::RBracket)?;
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::tokenize;

    #[test]
    fn test_parse_recall_with_type_filter() {
        let tokens = tokenize("RECALL memories WHERE type = episodic LIMIT 5").unwrap();
        let stmt = Parser::parse(&tokens).unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.filters.len(), 1);
                assert_eq!(r.filters[0].field, Field::Type);
                assert_eq!(r.filters[0].value, Value::MemoryType(MemoryType::Episodic));
                assert_eq!(r.limit, Some(5));
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn test_parse_recall_similar_to() {
        let tokens =
            tokenize(r#"RECALL memories WHERE content ~> "database migration" LIMIT 10"#).unwrap();
        let stmt = Parser::parse(&tokens).unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.filters.len(), 1);
                assert_eq!(r.filters[0].op, Operator::SimilarTo);
                assert_eq!(r.limit, Some(10));
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn test_parse_recall_near() {
        let tokens = tokenize("RECALL memories NEAR [0.1, 0.2, 0.3] LIMIT 10").unwrap();
        let stmt = Parser::parse(&tokens).unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.near, Some(vec![0.1, 0.2, 0.3]));
                assert_eq!(r.limit, Some(10));
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn test_parse_relate() {
        let tokens = tokenize(
            "RELATE 550e8400-e29b-41d4-a716-446655440000 -> 660e8400-e29b-41d4-a716-446655440000 AS caused WITH weight = 0.9"
        ).unwrap();
        let stmt = Parser::parse(&tokens).unwrap();
        match stmt {
            Statement::Relate(r) => {
                assert_eq!(r.edge_type, EdgeType::Caused);
                assert_eq!(r.weight, Some(0.9));
            }
            _ => panic!("expected Relate"),
        }
    }

    #[test]
    fn test_parse_forget() {
        let tokens = tokenize("FORGET 550e8400-e29b-41d4-a716-446655440000").unwrap();
        let stmt = Parser::parse(&tokens).unwrap();
        match stmt {
            Statement::Forget(f) => {
                assert_eq!(
                    f.target,
                    "550e8400-e29b-41d4-a716-446655440000".parse::<Uuid>().unwrap()
                );
            }
            _ => panic!("expected Forget"),
        }
    }

    #[test]
    fn test_parse_consolidate() {
        let tokens =
            tokenize(r#"CONSOLIDATE WHERE type = episodic AND accessed < "2024-01-01""#).unwrap();
        let stmt = Parser::parse(&tokens).unwrap();
        match stmt {
            Statement::Consolidate(c) => {
                assert_eq!(c.filters.len(), 2);
            }
            _ => panic!("expected Consolidate"),
        }
    }

    #[test]
    fn test_parse_traverse() {
        let tokens =
            tokenize("TRAVERSE 550e8400-e29b-41d4-a716-446655440000 DEPTH 3 WHERE edge_type = caused")
                .unwrap();
        let stmt = Parser::parse(&tokens).unwrap();
        match stmt {
            Statement::Traverse(t) => {
                assert_eq!(t.depth, 3);
                assert_eq!(t.edge_filter, Some(vec![EdgeType::Caused]));
            }
            _ => panic!("expected Traverse"),
        }
    }
}
