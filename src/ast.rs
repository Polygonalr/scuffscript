use crate::lexer::Token;

pub type GlobalId = String;
pub type VarId = String;
pub type NodeIdx = usize;

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub enum Binop {
    Add,
    Sub,
    Mul,
    // Eq, // This and below to be implemented when comparators and booleans are done
    // Neq,
    // Lt,
    // Lte,
    // Gt,
    // Gte,
    And,
    Or,
}

impl From<Token> for Binop {
    fn from(token: Token) -> Binop {
        match token {
            Token::Plus => Binop::Add,
            Token::Minus => Binop::Sub,
            Token::Times => Binop::Mul,
            Token::Ampersand => Binop::And,
            Token::Bar => Binop::Or,
            _ => panic!("Invalid token to Binop conversion")
        }
    }
}

impl ToString for Binop {
    fn to_string(&self) -> String {
        match self {
            Binop::Add => "+",
            Binop::Sub => "-",
            Binop::Mul => "*",
            Binop::And => "&",
            Binop::Or => "|",
        }.to_owned()
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub enum Term {
    Int(i64),
    Double(f64),
    Var(VarId),
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub enum Expr {
    // BinOp, Expr, Expr
    BinopExp(Binop, NodeIdx, NodeIdx),
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct FDecl {
    pub identifier: GlobalId,
    pub params: Vec<VarId>, // TODO Params should also include types
    // return_type:, // TODO For now everything is 64 bit integers
    pub statements: Vec<NodeIdx>,
}

impl FDecl {
    pub fn new(identifier: GlobalId, params: Vec<VarId>, statements: Vec<NodeIdx>) -> FDecl {
        FDecl {
            identifier,
            params,
            statements,
        }
    }

}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub enum Stmt {
    // VarId, Expr - LHS should be Expr when we add pointers jank
    Assn(VarId, NodeIdx),
    // VarId, Expr
    VDecl(VarId, NodeIdx),
    // Expr
    Ret(NodeIdx),
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub enum ASTNodeKind {
    StmtKind(Stmt),
    ExprKind(Expr),
    TermKind(Term),
    Empty,
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ASTNode {
    pub index: NodeIdx,
    pub kind: ASTNodeKind,
}

impl ASTNode {
    pub fn new(index: NodeIdx, kind: ASTNodeKind) -> ASTNode {
        ASTNode { index, kind }
    }
}

pub trait ASTStore {
    fn node_to_string(&self, idx: usize) -> String;
    fn stmt_to_string(&self, stmt: &Stmt) -> String;
    fn expr_to_string(&self, expr: &Expr) -> String;
    fn term_to_string(&self, term: &Term) -> String;
}

impl ASTStore for Vec<ASTNode> {
    fn node_to_string(&self, idx: usize) -> String {
        if idx >= self.len() {
            panic!("ASTStore::print_node: Index out of bounds");
        }
        let node = &self[idx];
        match &node.kind {
            ASTNodeKind::StmtKind(stmt) => {
                self.stmt_to_string(stmt)
            },
            ASTNodeKind::ExprKind(expr) => {
                self.expr_to_string(expr)
            },
            ASTNodeKind::TermKind(term) => {
                self.term_to_string(term)
            }
            _ => todo!()
        }
    }

    fn stmt_to_string(&self, stmt: &Stmt) -> String {
        match stmt {
            Stmt::Assn(vid, node_idx) => {
                format!("{} = {};", vid, self.node_to_string(*node_idx))
            },
            Stmt::VDecl(vid, node_idx) => {
                format!("let {} = {};", vid, self.node_to_string(*node_idx))
            },
            Stmt::Ret(node_idx) => {
                format!("return {};", self.node_to_string(*node_idx))
            },
        }
    }

    fn expr_to_string(&self, expr: &Expr) -> String {
        match expr {
            Expr::BinopExp(binop, lhs, rhs) => {
                format!("( {} {} {} )", self.node_to_string(*lhs), binop.to_string(), self.node_to_string(*rhs))
            }
        }
    }

    fn term_to_string(&self, term: &Term) -> String {
        match term {
            Term::Int(i) => {
                i.to_string()
            },
            Term::Double(d) => {
                d.to_string()
            },
            Term::Var(vid) => {
                vid.to_string()
            }
        }
    }
}
