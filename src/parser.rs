use std::fmt;
use std::iter::Peekable;

use crate::ast::{ASTNode, ASTNodeKind, ASTStore, Binop, Expr, FDecl, NodeIdx, Stmt, Term, VarId};
use crate::lexer::Token;

#[derive(Debug, Clone)]
pub struct ParserError {
    msg: String,
}

impl<'a> fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Parser Error: {}", self.msg)
    }
}

pub struct Parser {
    tokens: Peekable<std::vec::IntoIter<Token>>,
    ast_store: Vec<ASTNode>,
    func_decls: Vec<FDecl>,
}

impl From<Vec<Token>> for Parser {
    fn from(tokens: Vec<Token>) -> Parser {
        Parser {
            tokens: tokens.into_iter().peekable(),
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
            while let Some(param) = it.next() {
                curr.push(format!("{}", param));
                if it.peek().is_some() {
                    curr.push(", ".to_string());
                }
            }
            curr.push("], statements=[\n".to_string());
            for stmt_idx in func_decl.statements.iter() {
                curr.push(format!("  Stmt: {}\n", self.ast_store.node_to_string(*stmt_idx)));
            }
            curr.push("])".to_string());
            lines.push(curr.join(""))
        }
        lines.join("\n\n")
    }
}

impl Parser {
    /* ========= Utils ========= */
    fn expect(&mut self, expected: Token) -> Result<(), ParserError> {
        match self.tokens.peek() {
            None => Err(ParserError { msg: format!("Expected token {:?}, but buffer has no more tokens.", expected) }),
            Some(token) => {
                if *token != expected {
                    return Err(ParserError { msg: format!("Expected token {:?}, read {:?} instead", expected, token) });
                }
                Ok(())
            }
        }
    }

    fn expect_term(&mut self) -> Result<(), ParserError> {
        match self.tokens.peek() {
            None => Err(ParserError { msg: "Expected a terminal, but buffer has no more tokens.".to_string()}),
            Some(token) => {
                if !token.is_term() {
                    return Err(ParserError {
                        msg: format!("Expected a terminal, read {:?} instead", token)
                    });
                }
                Ok(())
            }
        }
    }

    fn peek(&mut self) -> Option<&Token> {
        self.tokens.peek()
    }

    fn consume(&mut self) -> Option<Token> {
        self.tokens.next()
    }

    fn expect_and_consume(&mut self, expected: Token) -> Result<Option<Token>, ParserError> {
        self.expect(expected)?;
        Ok(self.consume())
    }

    fn alloc_new_ast_idx(&mut self) -> usize {
        let new_idx = self.ast_store.len();
        self.ast_store.push(ASTNode::new(new_idx, ASTNodeKind::Empty));
        return new_idx;
    }

    fn infix_binding_power(op: &Token) -> (u8, u8) {
        match op {
            Token::Bar => (1, 2),
            // TODO Add XOR for (3, 4)
            Token::Ampersand => (5, 6),
            Token::Plus | Token::Minus => (7, 8),
            Token::Times => (9, 10),
            _ => (0, 0),
        }
    }

    /* ========= Beginning of recursive descent parsing ========= */

    pub fn parse_global(&mut self) -> Result<(), ParserError> {
        /* Returns nothing on success as this function will decide whether to push the parsed ASTNode to either the
        function declarations or the global statements. For now, only parse function declarations */
        match self.peek() {
            Some(Token::Function) => {
                let fdecl = self.parse_fdecl()?;
                self.func_decls.push(fdecl);
                return Ok(());
            }
            _ => {
                return Err(ParserError {
                    msg: "test3".to_string(),
                })
            }
        }
    }

    fn parse_fdecl(&mut self) -> Result<FDecl, ParserError> {
        self.expect_and_consume(Token::Function)?;

        let identifier = match self.consume() {
            Some(Token::Identifier(id)) => id,
            Some(stuff) => {
                return Err(ParserError {
                    msg: format!("{:?}", stuff),
                })
            },
            None => {
                return Err(ParserError {
                    msg: "nothing".to_string()
                })
            }
        };

        match self.consume() {
            Some(Token::LParen) => (),
            Some(token) => {
                return Err(ParserError {
                    msg: format!("Expected '(' in function declaration, got {:?} instead.", token),
                })
            },
            None => {
                return Err(ParserError {
                    msg: format!("Expected '(' in function declaration, but buffer has no more tokens."),
                })
            }
        }

        /*
         * Parse function arguments.
         * - If the next token is a RParen, then there are no arguments.
         * - If the next token is a type, then the next token should be an identifier.
         * - If the next token is an identifier, then the next token should be a comma or a RParen.
         * - If the next token is a comma, then the next token should be a type.
         * Informal Grammar:
         * args -> (type identifier (, type identifier)*)
         * type -> int | double | char | string
         * identifier -> [a-zA-Z_][a-zA-Z0-9_]*
         */
        #[derive(PartialEq)]
        enum ArgParseExpect {
            Type,          // Expecting a type following a comma
            Identifier,    // Expecting an identifier following a type
            CommaOrRParen, // Expecting a comma or right paran following an identifier
        }
        let mut params: Vec<VarId> = vec![];
        let peek = self.peek().unwrap().clone();
        if peek != Token::RParen {
            let mut arg_parse_expect = ArgParseExpect::Type;
            let mut is_closed_properly = false;
            while let Some(token) = self.consume() {
                match (token, arg_parse_expect) {
                    (t, ArgParseExpect::Type) => {
                        if t.is_type() {
                            arg_parse_expect = ArgParseExpect::Identifier;
                        } else {
                            return Err(ParserError {
                                msg: format!("Expected type in function declaration of {}, got {:?} instead.", identifier, t),
                            });
                        }
                    },
                    (Token::Identifier(arg_id), ArgParseExpect::Identifier) => {
                        params.push(arg_id.to_string());
                        arg_parse_expect = ArgParseExpect::CommaOrRParen;
                    }
                    (Token::Comma, ArgParseExpect::CommaOrRParen) => {
                        arg_parse_expect = ArgParseExpect::Type;
                    }
                    (Token::RParen, ArgParseExpect::CommaOrRParen) => {
                        is_closed_properly = true;
                        break;
                    }
                    _ => {
                        return Err(ParserError {
                            msg: "noarg2".to_string(),
                        })
                    }
                }
            }
            if !is_closed_properly {
                // No matching RParen to LParen
                return Err(ParserError {
                    msg: format!("Function arguments for {} not closed properly", identifier),
                });
            }
        } else {
            self.consume(); // Consume the RParen
        }

        let statements = self.parse_block()?;

        let new_fdecl = FDecl::new(identifier, params, statements);
        return Ok(new_fdecl);
    }

    fn parse_block(&mut self) -> Result<Vec<NodeIdx>, ParserError> {
        self.expect_and_consume(Token::LCurly)?;
        let mut statements: Vec<NodeIdx> = vec![];
        while self.peek() != Some(&Token::RCurly) {
            let stmt_node_idx = self.parse_stmt()?;
            statements.push(stmt_node_idx);
        }
        return Ok(statements);
    }

    fn parse_stmt(&mut self) -> Result<NodeIdx, ParserError> {
        let new_idx = match self.consume() {
            Some(Token::Let) => {
                let id = match self.consume() {
                    Some(Token::Identifier(id)) => id.clone(),
                    _ => {
                        return Err(ParserError {
                            msg: "".to_string(),
                        })
                    }
                };
                self.expect_and_consume(Token::Equal)?;
                let exp_node_idx = self.parse_exp(0)?;
                let ast_node_data =
                    ASTNodeKind::StmtKind(Stmt::VDecl(id.to_string(), exp_node_idx));
                let new_idx = self.alloc_new_ast_idx();
                self.ast_store[new_idx].update(ast_node_data);
                Ok(new_idx)
            }
            Some(Token::Identifier(id)) => {
                self.expect_and_consume(Token::Equal)?;
                let exp_node_idx = self.parse_exp(0)?;
                let ast_node_data = ASTNodeKind::StmtKind(Stmt::Assn(id.to_string(), exp_node_idx));
                let new_idx = self.alloc_new_ast_idx();
                self.ast_store[new_idx].update(ast_node_data);
                Ok(new_idx)
            }
            Some(Token::Return) => {
                let exp_node_idx = self.parse_exp(0)?;
                let ast_node_data = ASTNodeKind::StmtKind(Stmt::Ret(exp_node_idx));
                let new_idx = self.alloc_new_ast_idx();
                self.ast_store[new_idx].update(ast_node_data);
                Ok(new_idx)
            }
            Some(token) => Err(ParserError {
                msg: format!("Trying to parse start of statement, got {:?}", token)
            }),
            None => Err(ParserError{
                msg: "Trying to parse start of statement, but buffer has no more tokens.".to_string()
            })
        };
        self.expect_and_consume(Token::Semicolon)?;
        new_idx
    }

    /// 
    fn parse_exp(&mut self, min_priority: u8) -> Result<NodeIdx, ParserError> {
        let mut lhs_node_idx = match self.peek() {
            Some(Token::LParen) => {
                 self.consume();
                 let lhs = self.parse_exp(0);
                 self.expect_and_consume(Token::RParen)?;
                 lhs?
            },
            Some(_) => self.parse_term()?,
            None => return Err(ParserError { msg: "Trying to parse start of expression, but buffer has no more tokens.".to_string() }),
        };

        loop {
            let next_peek = self.peek();
            let op = match next_peek {
                Some(Token::Plus) | Some(Token::Minus) | Some(Token::Times) => {
                    next_peek.unwrap().clone()
                }
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
                ASTNodeKind::ExprKind(Expr::BinopExp(ast_op, lhs_node_idx, rhs_node_idx));
            let new_idx = self.alloc_new_ast_idx();
            self.ast_store[new_idx].update(ast_node_data);
            lhs_node_idx = new_idx;
        }

        Ok(lhs_node_idx)
    }

    fn parse_term(&mut self) -> Result<NodeIdx, ParserError> {
        /* Side note: checking whether a variable exists in a given context happens in the translation stage.
         * For now just generate the AST even if a variable does not exist in the current scope. */
        self.expect_term()?;
        let ast_node_data = match self.consume() {
            Some(Token::Identifier(id)) => ASTNodeKind::TermKind(Term::Var(id.to_string())),
            Some(Token::Integer(val)) => ASTNodeKind::TermKind(Term::Int(val)),
            Some(Token::Double(val)) => ASTNodeKind::TermKind(Term::Double(val)),
            _ => unreachable!(),
        };
        let new_idx = self.alloc_new_ast_idx();
        self.ast_store[new_idx].update(ast_node_data);
        return Ok(new_idx);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn parser_tests() {
        use crate::lexer::tokenize;
        use crate::parser::Parser;
        {
            let tokens = tokenize("function main () { let x = (40+2)*3; return x*2+(3-4)*5+6; }").unwrap();
            let mut parser = Parser::from(tokens);
            parser.parse_global().unwrap();
            assert_eq!(parser.to_string(), "FDecl(main, params=[], statements=[\n  Stmt: let x = ( ( 40 + 2 ) * 3 );\n  Stmt: return ( ( ( x * 2 ) + ( ( 3 - 4 ) * 5 ) ) + 6 );\n])");
        }
    }
}
