use std::fmt;
use std::iter::Peekable;

use crate::ast::{ASTNode, ASTNodeKind, ASTStore, Binop, Expr, FDecl, NodeIdx, Stmt, Term, VarId};
use crate::lexer::Token;

#[derive(Debug, Clone)]
pub struct ParserError {
    msg: String,
}

impl fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Parser Error: {}", self.msg) // TODO add location of error
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
                curr.push(param.to_owned());
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

    fn expect_and_consume(&mut self, expected: Token) -> Result<Option<Token>, ParserError> {
        self.expect(expected)?;
        Ok(self.tokens.next())
    }

    fn new_ast_node(&mut self, data: ASTNodeKind) -> usize {
        let new_idx = self.ast_store.len();
        self.ast_store.push(ASTNode::new(new_idx, data));
        new_idx
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

    /// Returns nothing on success as this function will decide whether to push the parsed ASTNode to either the
    /// function declarations or the global statements.
    /// 
    /// For now, only parse function declarations. Will parse other global declarations like struct definitions in the
    /// future.
    pub fn parse_global(&mut self) -> Result<(), ParserError> {
        match self.tokens.peek() {
            Some(Token::Function) => {
                let fdecl = self.parse_fdecl()?;
                self.func_decls.push(fdecl);
                Ok(())
            },
            Some(t) => {
                Err(ParserError {
                    msg: format!("Expected function keyword, got {:?} instead.", t),
                })
            },
            None => {
                Err(ParserError {
                    msg: "Expected function keyword, but buffer has no more tokens.".to_string(),
                })
            }
        }
    }

    /// Parse function declarations.
    /// 
    /// Grammar:
    /// - fdecl -> FUNCTION identifier params COLON type block
    /// - params -> LPAREN type identifier (COMMA type identifier)* RPAREN
    /// - type -> INTT | DOUBLET | CHART | STRINGT | BOOLT
    fn parse_fdecl(&mut self) -> Result<FDecl, ParserError> {
        self.expect_and_consume(Token::Function)?;

        let identifier = match self.tokens.next() {
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

        match self.tokens.next() {
            Some(Token::LParen) => (),
            Some(token) => {
                return Err(ParserError {
                    msg: format!("Expected '(' in function declaration, got {:?} instead.", token),
                })
            },
            None => {
                return Err(ParserError {
                    msg: "Expected '(' in function declaration, but buffer has no more tokens.".to_string(),
                })
            }
        }

        /*
         * Parse function parameters.
         * - If the next token is a RParen, then there are no parameters.
         * - If the next token is a type, then the next token should be an identifier.
         * - If the next token is an identifier, then the next token should be a comma or a RParen.
         * - If the next token is a comma, then the next token should be a type.
         */
        #[derive(PartialEq)]
        enum ArgParseExpect {
            Type,          // Expecting a type following a comma
            Identifier,    // Expecting an identifier following a type
            CommaOrRParen, // Expecting a comma or right paran following an identifier
        }
        let mut params: Vec<VarId> = vec![];
        let peek = self.tokens.peek().unwrap().clone();
        if peek != Token::RParen {
            let mut arg_parse_expect = ArgParseExpect::Type;
            let mut is_closed_properly = false;
            while let Some(token) = self.tokens.next() {
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
            self.tokens.next(); // Consume the RParen
        }

        let statements = self.parse_block()?;

        Ok(FDecl::new(identifier, params, statements))
    }

    /// Parse a block of statements.
    /// 
    /// Grammar:
    /// - block -> LCURLY { statement_1 ... statement_n } RCURLY
    fn parse_block(&mut self) -> Result<Vec<NodeIdx>, ParserError> {
        self.expect_and_consume(Token::LCurly)?;
        let mut statements: Vec<NodeIdx> = vec![];
        while self.tokens.peek() != Some(&Token::RCurly) {
            let stmt_node_idx = self.parse_stmt()?;
            statements.push(stmt_node_idx);
        }
        Ok(statements)
    }

    /// Parse a statement.
    /// 
    /// Grammar (<THIS> stands for in-code naming):
    /// - statement -> ( variable_declaration <VDecl> | assignment <Assn> | return_stmt <Ret> ) SEMICOLON
    /// - variable_declaration -> LET identifier = expression
    /// - assignment -> identifier = expression
    /// - return_stmt -> RETURN expression
    fn parse_stmt(&mut self) -> Result<NodeIdx, ParserError> {
        let new_idx = match self.tokens.next() {
            Some(Token::Let) => {
                let id = match self.tokens.next() {
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
                Ok(self.new_ast_node(ast_node_data))
            }
            Some(Token::Identifier(id)) => {
                self.expect_and_consume(Token::Equal)?;
                let exp_node_idx = self.parse_exp(0)?;
                let ast_node_data = ASTNodeKind::StmtKind(Stmt::Assn(id.to_string(), exp_node_idx));
                Ok(self.new_ast_node(ast_node_data))
            }
            Some(Token::Return) => {
                let exp_node_idx = self.parse_exp(0)?;
                let ast_node_data = ASTNodeKind::StmtKind(Stmt::Ret(exp_node_idx));
                Ok(self.new_ast_node(ast_node_data))
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

    /// Parse an expression. Pratt parsing is used here to deal with precedences.
    /// (reference: https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html)
    /// 
    /// TODO: Add support for prefix operator.
    /// 
    /// Grammar
    /// - expression -> terminal | terminal binop expression
    /// - binop (<PRECEDENCE>) -> BAR <1> | CARET <2> | AMPERSAND <3> | PLUS <4> | MINUS <4> | TIMES <5>
    fn parse_exp(&mut self, min_priority: u8) -> Result<NodeIdx, ParserError> {
        let mut lhs_node_idx = match self.tokens.peek() {
            Some(Token::LParen) => {
                 self.tokens.next();
                 let lhs = self.parse_exp(0);
                 self.expect_and_consume(Token::RParen)?;
                 lhs?
            },
            Some(_) => self.parse_term()?,
            None => return Err(ParserError { msg: "Trying to parse start of expression, but buffer has no more tokens.".to_string() }),
        };

        loop {
            let next_peek = self.tokens.peek();
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
            self.tokens.next();
            let rhs_node_idx = self.parse_exp(right_op_priority)?;
            let ast_op = Binop::from(op.clone());
            let ast_node_data =
                ASTNodeKind::ExprKind(Expr::BinopExp(ast_op, lhs_node_idx, rhs_node_idx));
            lhs_node_idx = self.new_ast_node(ast_node_data);
        }

        Ok(lhs_node_idx)
    }

    /// Parse a terminal.
    /// 
    /// Grammar
    /// - terminal -> identifier | integer | double | string | true | false (these definitions are handled by lexer)
    fn parse_term(&mut self) -> Result<NodeIdx, ParserError> {
        /* Side note: checking whether a variable exists in a given context happens in the translation stage.
         * For now just generate the AST even if a variable does not exist in the current scope. */
        self.expect_term()?;
        let ast_node_data = match self.tokens.next() {
            Some(Token::Identifier(id)) => ASTNodeKind::TermKind(Term::Var(id.to_string())),
            Some(Token::Integer(val)) => ASTNodeKind::TermKind(Term::Int(val)),
            Some(Token::Double(val)) => ASTNodeKind::TermKind(Term::Double(val)),
            _ => unreachable!(),
        };
        Ok(self.new_ast_node(ast_node_data))
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
