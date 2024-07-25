use crate::lexer::{SourceLocation, TokenData};

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

impl From<TokenData> for Binop {
    fn from(token: TokenData) -> Binop {
        match token {
            TokenData::Plus => Binop::Add,
            TokenData::Minus => Binop::Sub,
            TokenData::Times => Binop::Mul,
            TokenData::Ampersand => Binop::And,
            TokenData::Bar => Binop::Or,
            _ => panic!("Invalid token to Binop conversion"),
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
        }
        .to_owned()
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub enum Type {
    Int,
    Double,
    Bool,
    String,
    Void,
}

impl ToString for Type {
    fn to_string(&self) -> String {
        match self {
            Type::Int => "int".to_string(),
            Type::Double => "double".to_string(),
            Type::Bool => "bool".to_string(),
            Type::String => "string".to_string(),
            Type::Void => "void".to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub enum Term {
    Int(i64),
    Double(f64),
    Var(VarId),
    FuncCall(GlobalId, Vec<NodeIdx>),
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub enum Expr {
    // BinOp, Expr, Expr
    BinopExp(Binop, NodeIdx, NodeIdx),
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct FDecl {
    pub identifier: GlobalId,
    pub params: Vec<(VarId, Type)>, // TODO Params should also include types
    // return_type:, // TODO For now everything is 64 bit integers
    pub statements: Vec<NodeIdx>,
}

impl FDecl {
    pub fn new(identifier: GlobalId, params: Vec<(VarId, Type)>, statements: Vec<NodeIdx>) -> FDecl {
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
    Stmt(Stmt),
    Expr(Expr),
    Term(Term),
}

#[derive(Clone, Debug)]
pub struct ASTNode {
    pub index: NodeIdx,
    pub kind: ASTNodeKind,
    pub loc: SourceLocation,
}

impl ASTNode {
    pub fn new(index: NodeIdx, kind: ASTNodeKind, loc: SourceLocation) -> ASTNode {
        ASTNode { index, kind, loc }
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
            ASTNodeKind::Stmt(stmt) => self.stmt_to_string(stmt),
            ASTNodeKind::Expr(expr) => self.expr_to_string(expr),
            ASTNodeKind::Term(term) => self.term_to_string(term),
        }
    }

    fn stmt_to_string(&self, stmt: &Stmt) -> String {
        match stmt {
            Stmt::Assn(vid, node_idx) => {
                format!("{} = {};", vid, self.node_to_string(*node_idx))
            }
            Stmt::VDecl(vid, node_idx) => {
                format!("let {} = {};", vid, self.node_to_string(*node_idx))
            }
            Stmt::Ret(node_idx) => {
                format!("return {};", self.node_to_string(*node_idx))
            }
        }
    }

    fn expr_to_string(&self, expr: &Expr) -> String {
        match expr {
            Expr::BinopExp(binop, lhs, rhs) => {
                format!(
                    "( {} {} {} )",
                    self.node_to_string(*lhs),
                    binop.to_string(),
                    self.node_to_string(*rhs)
                )
            }
        }
    }

    fn term_to_string(&self, term: &Term) -> String {
        match term {
            Term::Int(i) => i.to_string(),
            Term::Double(d) => d.to_string(),
            Term::Var(vid) => vid.to_string(),
            Term::FuncCall(id, args) => {
                let args_str = args
                    .iter()
                    .map(|arg_idx| self.node_to_string(*arg_idx))
                    .collect::<Vec<String>>()
                    .join(", ");
                format!("{}({})", id, args_str)
            }
        }
    }
}
