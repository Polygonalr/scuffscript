use anyhow::{anyhow, Result};
use std::fmt;
use std::iter::Peekable;
use thiserror::Error;

use crate::ast::{
    ASTNode, ASTNodeKind, ASTStore, Binop, Expr, FDecl, NodeIdx, Stmt, Term, Type, VarId,
};
use crate::lexer::{SourceLocation, Token, TokenData};

#[derive(Error, Debug, Clone)]
pub struct ParserError {
    loc: SourceLocation,
    msg: String,
}

impl fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Parser Error at {} - {}", self.loc.to_string(), self.msg)
    }
}

/// The parser. Takes in a list of tokens and outputs an AST.
///
/// An AST contains the `func_decls` vector which contains all function declarations, alongside with all the nodes that
/// are present in the AST within `ast_store`. The data structore of this AST is inspired by Zig compiler's. Reference
/// for it here: https://mitchellh.com/zig/parser.
///
/// Parser is hand-written like other mainstream compilers in recursive descent style, mainly for my own practice and to
/// handle error messages easier.
pub struct Parser {
    /// Iterator of tokens to be consumed by the parser. Peekable so that the grammar is at least LL(1).
    tokens: Peekable<std::vec::IntoIter<Token>>,
    last_consumed_loc: SourceLocation,
    pub ast_store: Vec<ASTNode>,
    pub func_decls: Vec<FDecl>,
}

impl From<Vec<Token>> for Parser {
    fn from(tokens: Vec<Token>) -> Parser {
        Parser {
            tokens: tokens.into_iter().peekable(),
            last_consumed_loc: SourceLocation {
                source_path: "".into(),
                line: 0,
                col: 0,
            },
            ast_store: vec![],
            func_decls: vec![],
        }
    }
}

impl ToString for Parser {
    fn to_string(&self) -> String {
        let mut lines: Vec<String> = vec![];
        for func_decl in self.func_decls.iter() {
            let mut curr: Vec<String> = vec![];
            curr.push(format!("FDecl({}, params=[", func_decl.identifier));
            let mut it = func_decl.params.iter().peekable();
            while let Some((param_id, param_type)) = it.next() {
                curr.push(format!(
                    "{} : {}",
                    param_id.to_owned(),
                    param_type.to_string()
                ));
                if it.peek().is_some() {
                    curr.push(", ".to_string());
                }
            }
            curr.push("], statements=[\n".to_string());
            for stmt_idx in func_decl.statements.iter() {
                curr.push(format!(
                    "  Stmt: {}\n",
                    self.ast_store.node_to_string(*stmt_idx)
                ));
            }
            curr.push("])".to_string());
            lines.push(curr.join(""))
        }
        lines.join("\n\n")
    }
}

impl Parser {
    /* ========= Utils ========= */
    fn new_err(&self, msg: String) -> anyhow::Error {
        anyhow::Error::new(ParserError {
            msg,
            loc: self.last_consumed_loc.clone(),
        })
    }

    fn consume(&mut self) -> Option<TokenData> {
        match self.tokens.next() {
            Some(token) => {
                self.last_consumed_loc = token.loc; // Update cached loc for error message
                Some(token.data)
            }
            None => None,
        }
    }

    fn peek_token_data(&mut self) -> Option<TokenData> {
        self.tokens.peek().map(|token| token.data.clone())
    }

    fn expect(&mut self, expected: TokenData) -> Result<()> {
        match self.peek_token_data() {
            None => Err(anyhow!(self.new_err(format!(
                "Expected token {:?}, but buffer has no more tokens.",
                expected
            )))),
            Some(token) => {
                if token != expected {
                    return Err(anyhow!(self.new_err(format!(
                        "Expected token {:?}, read {:?} instead",
                        expected, token
                    ))));
                }
                Ok(())
            }
        }
    }

    fn expect_term(&mut self) -> Result<()> {
        match self.peek_token_data() {
            None => {
                Err(self.new_err("Expected a terminal, but buffer has no more tokens.".to_string()))
            }
            Some(token) => {
                if !token.is_term() {
                    return Err(
                        self.new_err(format!("Expected a terminal, read {:?} instead", token))
                    );
                }
                Ok(())
            }
        }
    }

    fn expect_type(&mut self) -> Result<()> {
        match self.peek_token_data() {
            None => {
                Err(self.new_err("Expected a type, but buffer has no more tokens.".to_string()))
            }
            Some(token) => {
                if !token.is_type() {
                    return Err(self.new_err(format!("Expected a type, read {:?} instead", token)));
                }
                Ok(())
            }
        }
    }

    fn expect_and_consume(&mut self, expected: TokenData) -> Result<Option<TokenData>> {
        self.expect(expected)?;
        Ok(self.consume())
    }

    fn new_ast_node(&mut self, data: ASTNodeKind) -> usize {
        let new_idx = self.ast_store.len();
        self.ast_store
            .push(ASTNode::new(new_idx, data, self.last_consumed_loc.clone()));
        new_idx
    }

    fn infix_binding_power(op: &TokenData) -> (u8, u8) {
        match op {
            TokenData::Bar => (1, 2),
            // TODO Add XOR for (3, 4)
            TokenData::Ampersand => (5, 6),
            TokenData::Plus | TokenData::Minus => (7, 8),
            TokenData::Times => (9, 10),
            _ => (0, 0),
        }
    }

    pub fn parse(&mut self) -> Result<()> {
        while self.peek_token_data().is_some() {
            if let Some(token) = self.peek_token_data() {
                if token == TokenData::Eof {
                    break;
                }
            }
            self.parse_global()?;
        }
        Ok(())
    }

    /* ========= Beginning of recursive descent parsing ========= */

    /// Returns nothing on success as this function will decide whether to push the parsed ASTNode to either the
    /// function declarations or the global statements.
    ///
    /// For now, only parse function declarations. Will parse other global declarations like struct definitions in the
    /// future.
    fn parse_global(&mut self) -> Result<()> {
        match self.peek_token_data() {
            Some(TokenData::Function) => {
                let fdecl = self.parse_fdecl()?;
                self.func_decls.push(fdecl);
                Ok(())
            }
            Some(t) => {
                Err(self.new_err(format!("Expected function keyword, got {:?} instead.", t)))
            }
            None => Err(self
                .new_err("Expected function keyword, but buffer has no more tokens.".to_string())),
        }
    }

    /// Parse function declarations.
    ///
    /// Grammar:
    /// - fdecl -> FUNCTION identifier params COLON type block
    /// - params -> LPAREN type identifier (COMMA type identifier)* RPAREN
    /// - type -> INTT | DOUBLET | CHART | STRINGT | BOOLT
    fn parse_fdecl(&mut self) -> Result<FDecl> {
        self.expect_and_consume(TokenData::Function)?;

        let identifier = match self.consume() {
            None => return Err(self.new_err(
                "Expected an identifier in function declaration, but buffer has no more tokens."
                    .to_string(),
            )),
            Some(TokenData::Identifier(id)) => id,
            Some(t) => {
                return Err(self.new_err(format!(
                    "Expected an identifier in function declaration, got {:?} instead",
                    t
                )))
            }
        };

        self.expect_and_consume(TokenData::LParen)?;

        /*
         * Parse function parameters.
         * - If the next token is a RParen, then there are no parameters.
         * - If the next token is a type, then the next token should be an identifier.
         * - If the next token is an identifier, then the next token should be a comma or a RParen.
         * - If the next token is a comma, then the next token should be a type.
         */
        #[derive(PartialEq)]
        enum ArgParseExpect {
            Argument,      // Expecting an argument declaraction in the form "id : type"
            CommaOrRParen, // Expecting a comma or right paran following an identifier
        }
        let mut params: Vec<(VarId, Type)> = vec![];

        if self.peek_token_data() != Some(TokenData::RParen) {
            let mut arg_parse_expect = ArgParseExpect::Argument;
            let mut is_closed_properly = false;
            while let Some(token_data) = self.consume() {
                match (token_data, arg_parse_expect) {
                    (TokenData::Identifier(arg_id), ArgParseExpect::Argument) => {
                        params.push((arg_id.to_string(), Type::Int)); // TODO For now everything is int
                        self.expect_and_consume(TokenData::Colon)?;
                        self.expect_type()?;
                        // TODO for now everything is int
                        self.expect_and_consume(TokenData::IntT)?;
                        arg_parse_expect = ArgParseExpect::CommaOrRParen;
                    }
                    (TokenData::Comma, ArgParseExpect::CommaOrRParen) => {
                        arg_parse_expect = ArgParseExpect::Argument;
                    }
                    (TokenData::RParen, ArgParseExpect::CommaOrRParen) => {
                        is_closed_properly = true;
                        break;
                    }
                    (t, _) => {
                        return Err(self.new_err(format!(
                            "Unexpected token {:?} found in the function declaration {}.",
                            t, identifier
                        )))
                    }
                }
            }
            if !is_closed_properly {
                // No matching RParen to LParen
                return Err(self.new_err(format!(
                    "Function arguments for {} not closed properly",
                    identifier
                )));
            }
        } else {
            self.consume(); // Consume the RParen
        }

        /* End of parsing function parameters */
        self.expect_and_consume(TokenData::Colon)?;
        self.expect_and_consume(TokenData::IntT)?; // TODO for now, expect all functions to return int

        let statements = self.parse_block()?;

        Ok(FDecl::new(identifier, params, statements))
    }

    /// Parse a block of statements. Blocks are represented as a vector of nodes.
    ///
    /// Grammar:
    /// - block -> LCURLY { statement_1 ... statement_n } RCURLY
    fn parse_block(&mut self) -> Result<Vec<NodeIdx>> {
        self.expect_and_consume(TokenData::LCurly)?;
        let mut statements: Vec<NodeIdx> = vec![];
        while self.peek_token_data() != Some(TokenData::RCurly) {
            if self.peek_token_data().is_none() {
                return Err(self.new_err(
                    "Function declaration code block does not have a matching '}'.".to_string(),
                ));
            }
            let stmt_node_idx = self.parse_stmt()?;
            statements.push(stmt_node_idx);
        }
        self.expect_and_consume(TokenData::RCurly)?;
        Ok(statements)
    }

    /// Parse a statement.
    ///
    /// Grammar (<THIS> stands for in-code naming):
    /// - statement -> ( variable_declaration <VDecl> | assignment <Assn> | return_stmt <Ret> ) SEMICOLON
    /// - variable_declaration -> LET identifier = expression
    /// - assignment -> identifier = expression
    /// - return_stmt -> RETURN expression
    fn parse_stmt(&mut self) -> Result<NodeIdx> {
        let mut expect_semicolon = true;
        let new_idx = match self.consume() {
            Some(TokenData::Let) => {
                let id = match self.consume() {
                    Some(TokenData::Identifier(id)) => id.clone(),
                    Some(t) => {
                        return Err(
                            self.new_err(format!("Expected an identifier, got {:?} instead.", t))
                        )
                    }
                    None => {
                        return Err(self.new_err(
                            "Expected identifier after 'let', but buffer has no more tokens."
                                .to_string(),
                        ))
                    }
                };
                self.expect_and_consume(TokenData::Equal)?;
                let exp_node_idx = self.parse_exp(0)?;
                let ast_node_data = ASTNodeKind::Stmt(Stmt::VDecl(id.to_string(), exp_node_idx));
                Ok(self.new_ast_node(ast_node_data))
            }
            Some(TokenData::Identifier(id)) => {
                self.expect_and_consume(TokenData::Equal)?;
                let exp_node_idx = self.parse_exp(0)?;
                let ast_node_data = ASTNodeKind::Stmt(Stmt::Assn(id.to_string(), exp_node_idx));
                Ok(self.new_ast_node(ast_node_data))
            }
            Some(TokenData::Return) => {
                let exp_node_idx = self.parse_exp(0)?;
                let ast_node_data = ASTNodeKind::Stmt(Stmt::Ret(exp_node_idx));
                Ok(self.new_ast_node(ast_node_data))
            }
            Some(TokenData::If) => {
                expect_semicolon = false;
                Ok(self.parse_if()?)
            }
            Some(token) => Err(self.new_err(format!(
                "Trying to parse start of statement, got {:?}",
                token
            ))),
            None => Err(self.new_err(
                "Trying to parse start of statement, but buffer has no more tokens.".to_string(),
            )),
        };
        if expect_semicolon {
            self.expect_and_consume(TokenData::Semicolon)?;
        }
        new_idx
    }

    /// Parse an expression. Pratt parsing is used here to deal with precedences.
    /// (reference: https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html)
    ///
    /// TODO: Add support for prefix operator.
    ///
    /// Grammar
    /// - expression -> terminal | terminal binop expression
    /// - binop (<PRECEDENCE>) -> BAR <1> | CARET <2> | AMPERSAND <3> | PLUS <4> | MINUS <4> | TIMES <5>
    fn parse_exp(&mut self, min_priority: u8) -> Result<NodeIdx> {
        let mut lhs_node_idx = match self.peek_token_data() {
            Some(TokenData::LParen) => {
                self.consume();
                let lhs = self.parse_exp(0);
                self.expect_and_consume(TokenData::RParen)?;
                lhs?
            }
            Some(_) => self.parse_term()?,
            None => {
                return Err(self.new_err(
                    "Trying to parse start of expression, but buffer has no more tokens."
                        .to_string(),
                ))
            }
        };

        loop {
            let next_peek = self.peek_token_data();
            let op = match next_peek {
                Some(TokenData::Plus)
                | Some(TokenData::Minus)
                | Some(TokenData::Times)
                | Some(TokenData::EqualEqual)
                | Some(TokenData::ExclEqual)
                | Some(TokenData::LAngle)
                | Some(TokenData::LAngleEqual)
                | Some(TokenData::RAngle)
                | Some(TokenData::RAngleEqual) => next_peek.unwrap().clone(),
                _ => break, // End of expression parsing
            };
            let (left_op_priority, right_op_priority) = Self::infix_binding_power(&op);
            if left_op_priority < min_priority {
                break;
            }
            self.consume();
            let rhs_node_idx = self.parse_exp(right_op_priority)?;
            let ast_op = Binop::from(op.clone());
            let ast_node_data =
                ASTNodeKind::Expr(Expr::BinopExp(ast_op, lhs_node_idx, rhs_node_idx));
            lhs_node_idx = self.new_ast_node(ast_node_data);
        }

        Ok(lhs_node_idx)
    }

    /// Parse a terminal.
    ///
    /// Grammar
    /// - terminal -> func_call | identifier | integer | double | string | true | false (these definitions are handled by lexer)
    fn parse_term(&mut self) -> Result<NodeIdx> {
        /* Side note: checking whether a variable exists in a given context happens in the translation stage.
         * For now just generate the AST even if a variable does not exist in the current scope. */
        self.expect_term()?;
        let next_token = self.consume().unwrap(); // Should not panic as expect_term is called beforehand.
        let ast_node_data =
            match next_token {
                TokenData::Identifier(id) => match self.peek_token_data() {
                    Some(TokenData::LParen) => return self.parse_func_call(id),
                    Some(_) => ASTNodeKind::Term(Term::Var(id.to_string())),
                    None => return Err(self.new_err(
                        "Expected something after an identifier, but buffer has no more tokens."
                            .to_string(),
                    )),
                },
                TokenData::Integer(val) => ASTNodeKind::Term(Term::Int(val)),
                TokenData::Double(val) => ASTNodeKind::Term(Term::Double(val)),
                _ => unreachable!(),
            };
        Ok(self.new_ast_node(ast_node_data))
    }

    fn parse_func_call(&mut self, id: String) -> Result<NodeIdx> {
        self.expect_and_consume(TokenData::LParen)?;
        let mut args: Vec<NodeIdx> = vec![];
        while self.peek_token_data() != Some(TokenData::RParen) {
            let exp_node_idx = self.parse_exp(0)?;
            args.push(exp_node_idx);
            match self.peek_token_data() {
                Some(TokenData::Comma) => {
                    self.consume();
                }
                Some(TokenData::RParen) => (),
                Some(token_data) => {
                    return Err(self.new_err(format!(
                        "Expected either ',' or ')', got {:?} instead.",
                        token_data
                    )))
                }
                None => {
                    return Err(self.new_err(
                        "Expected either ',' or ')', but buffer has no more tokens.".to_string(),
                    ))
                }
            }
        }
        self.expect_and_consume(TokenData::RParen)?;
        let ast_node_data = ASTNodeKind::Term(Term::FuncCall(id, args));
        Ok(self.new_ast_node(ast_node_data))
    }

    /// Parse an if-else block
    ///
    /// Grammar
    /// - if_block -> IF LPAREN expression RPAREN block (ELSE block)?
    fn parse_if(&mut self) -> Result<NodeIdx> {
        self.expect_and_consume(TokenData::LParen)?;
        let condition = self.parse_exp(0)?;
        self.expect_and_consume(TokenData::RParen)?;
        let if_block = self.parse_block()?;
        let else_block = match self.peek_token_data() {
            Some(TokenData::Else) => {
                self.consume();
                Some(self.parse_block()?)
            }
            _ => None,
        };
        let ast_node_data = ASTNodeKind::Stmt(Stmt::IfElse(condition, if_block, else_block));
        Ok(self.new_ast_node(ast_node_data))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn parser_simple_tests() {
        use crate::lexer::tokenize;
        use crate::parser::Parser;
        {
            let tokens =
                tokenize("function main () : int { let x = (40+2)*3; return x*2+(3-4)*5+6; }")
                    .unwrap();
            let mut parser = Parser::from(tokens);
            parser.parse_global().unwrap();
            assert_eq!(
                parser.to_string(),
                "\
                FDecl(main, params=[], statements=[\n  \
                Stmt: let x = ( ( 40 + 2 ) * 3 );\n  \
                Stmt: return ( ( ( x * 2 ) + ( ( 3 - 4 ) * 5 ) ) + 6 );\n])\
            "
            );
        }
        {
            let tokens = tokenize("function main () : int { let x = 3; return x; }").unwrap();
            let mut parser = Parser::from(tokens);
            parser.parse_global().unwrap();
            assert_eq!(
                parser.to_string(),
                "FDecl(main, params=[], statements=[\n  Stmt: let x = 3;\n  Stmt: return x;\n])"
            );
        }
    }

    #[test]
    fn parser_func_decl_tests() {
        use crate::lexer::tokenize;
        use crate::parser::Parser;
        {
            let tokens =
                tokenize("function add (a : int, b : int) : int { return a + b; }").unwrap();
            let mut parser = Parser::from(tokens);
            parser.parse_global().unwrap();
            assert_eq!(
                parser.to_string(),
                "FDecl(add, params=[a : int, b : int], statements=[\n  Stmt: return ( a + b );\n])"
            );
        }
    }

    #[test]
    fn parser_func_call_tests() {
        use crate::lexer::tokenize;
        use crate::parser::Parser;
        {
            let tokens = tokenize("function main () : int { return add(3, 4); }").unwrap();
            let mut parser = Parser::from(tokens);
            parser.parse_global().unwrap();
            assert_eq!(
                parser.to_string(),
                "FDecl(main, params=[], statements=[\n  Stmt: return add(3, 4);\n])"
            );
        }
        {
            let tokens = tokenize("function main () : int { return add(5*6+2, 4); }").unwrap();
            let mut parser = Parser::from(tokens);
            parser.parse_global().unwrap();
            assert_eq!(
                parser.to_string(),
                "FDecl(main, params=[], statements=[\n  Stmt: return add(( ( 5 * 6 ) + 2 ), 4);\n])"
            );
        }
    }

    #[test]
    fn parser_if_else_test() {
        use crate::lexer::tokenize;
        use crate::parser::Parser;
        {
            let tokens =
                tokenize("function main () : int { if (1) { return 1; } else { return 2; } }")
                    .unwrap();
            let mut parser = Parser::from(tokens);
            parser.parse_global().unwrap();
            assert_eq!(
                parser.to_string(),
                "FDecl(main, params=[], statements=[\n  Stmt: if ( 1 ) { return 1; } else { return 2; }\n])"
            );
        }
        {
            let tokens = tokenize("function main () : int { if (1) { return 1; } }").unwrap();
            let mut parser = Parser::from(tokens);
            parser.parse_global().unwrap();
            assert_eq!(
                parser.to_string(),
                "FDecl(main, params=[], statements=[\n  Stmt: if ( 1 ) { return 1; } \n])"
            );
        }
    }
}
