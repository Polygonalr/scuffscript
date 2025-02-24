use crate::ast::{Expr, Term};
use crate::lexer::SourceLocation;
use crate::mlir::ast::{MlirType, OpData, OperandId};
use crate::mlir::utils::MlirIdGen;
use crate::{
    ast::{ASTNode, ASTNodeKind, FDecl, Stmt, Type, VarId},
    mlir::ast::{Func, MlirProg, Op},
};
use std::collections::HashMap;
use std::rc::Rc;
use std::{fmt, mem};

#[derive(Debug, Clone)]
pub struct FrontendError {
    loc: SourceLocation,
    msg: String,
}

impl fmt::Display for FrontendError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Frontend Compilation Error at {} - {}",
            self.loc.to_string(),
            self.msg
        )
    }
}

struct VarMetadata {
    var_type: Type,
    operand_id: OperandId,
}

/// Context of the current function the frontend is currently translating.
struct Context {
    scope: HashMap<VarId, (VarMetadata, usize)>, // Variable identifier : (Variable metadata, scope depth)
    depth: usize,
    // TODO add vector of struct declarations and other global decls
}

/// Frontend store for temporary compilation context and the output program.
pub struct Frontend {
    ctxt: Context,
    id_gen: MlirIdGen,
    hoisted_ops: Vec<Op>,
    ops: Vec<Op>,
    ast_store: Rc<[ASTNode]>,
    func_decls: Rc<[FDecl]>,
}

impl Frontend {
    pub fn new(ast_store: Vec<ASTNode>, func_decls: Vec<FDecl>) -> Self {
        Self {
            ctxt: Context {
                scope: HashMap::new(),
                depth: 0,
            },
            id_gen: MlirIdGen::new(),
            hoisted_ops: Vec::new(),
            ops: Vec::new(),
            ast_store: Rc::from(ast_store),
            func_decls: Rc::from(func_decls),
        }
    }

    fn reset_ctxt(&mut self) {
        self.ctxt = {
            Context {
                scope: HashMap::new(),
                depth: 0,
            }
        }
    }

    pub fn compile_to_mlir(&mut self) -> Result<MlirProg, FrontendError> {
        let mut mlir_prog: MlirProg = MlirProg::new();

        // TODO compile & add global variables to context before compiling functions

        let func_decls = Rc::clone(&self.func_decls); // Rc clone here so that we won't borrow *self in the next line.
        let ast_store = Rc::clone(&self.ast_store);

        // compile func_decls
        for ast_func in func_decls.iter() {
            let mut mlir_func = Func::new(ast_func.identifier.clone());
            let curr_depth = 0;

            for (param_id, param_type) in &ast_func.params {
                // params are passed as raw values, we need to store them as local memrefs first
                let param_raw_op_id = self.id_gen.gen_id(&format!("param_{}", param_id));
                let memref_op_id = self.id_gen.gen_id(param_id);
                let mlir_type = MlirType::from(param_type);
                let alloca_op =
                    Op::new(Some(memref_op_id.clone()), OpData::MemrefAlloca(mlir_type));
                self.hoisted_ops.push(alloca_op);
                let store_op = Op::new(None, OpData::MemrefStore(memref_op_id.clone(), param_raw_op_id.clone()));
                self.hoisted_ops.push(store_op);

                let metadata = VarMetadata {
                    var_type: param_type.clone(),
                    operand_id: memref_op_id,
                };
                mlir_func.append_param((param_raw_op_id, MlirType::from(param_type)));

                self.ctxt
                    .scope
                    .insert(param_id.clone(), (metadata, curr_depth));
            }

            for statement_idx in &ast_func.statements {
                match &ast_store[*statement_idx].kind {
                    ASTNodeKind::Stmt(stmt) => self.compile_statement(stmt),
                    kind => panic!("Expected statement AST node but got {:?} instead.", kind),
                }?;
            }

            // Consume and reset hoisted_ops and ops
            let hoisted_ops = mem::take(&mut self.hoisted_ops);
            mlir_func.append_ops(hoisted_ops);
            let ops = mem::take(&mut self.ops);
            mlir_func.append_ops(ops);

            mlir_prog.add_func(mlir_func);

            self.reset_ctxt();
        }

        Ok(mlir_prog)
    }

    fn compile_statement(&mut self, stmt: &Stmt) -> Result<(), FrontendError> {
        match stmt {
            Stmt::Assn(var_id, node_idx) => self.compile_assn(var_id, *node_idx),
            Stmt::VDecl(var_id, node_idx) => self.compile_vdecl(var_id, *node_idx),
            Stmt::Ret(node_idx) => self.compile_ret(*node_idx),
            Stmt::IfElse(cond_expr_idx, if_block, else_block) => self.compile_ifelse(*cond_expr_idx, if_block, else_block),
        }
    }

    fn compile_assn(&mut self, var_id: &str, node_idx: usize) -> Result<(), FrontendError> {
        if !self.ctxt.scope.contains_key(var_id) {
            return Err(FrontendError {
                loc: self.ast_store[node_idx].loc.clone(),
                msg: format!("Variable {} not declared in scope", var_id),
            });
        }
        todo!()
    }

    fn compile_vdecl(&mut self, var_id: &str, node_idx: usize) -> Result<(), FrontendError> {
        if self.ctxt.scope.contains_key(var_id) {
            return Err(FrontendError {
                loc: self.ast_store[node_idx].loc.clone(),
                msg: format!("Variable {} already declared in scope", var_id),
            });
        }

        // Compile operations to allocate memory for new variable
        let operand_id = self.id_gen.gen_id(var_id);
        let metadata = VarMetadata {
            var_type: Type::Int,
            operand_id: operand_id.clone(),
        };

        self.ctxt
            .scope
            .insert(var_id.to_string(), (metadata, self.ctxt.depth));
        let alloca_op = Op::new(
            Some(operand_id.clone()),
            OpData::MemrefAlloca(MlirType::I64),
        );
        self.hoisted_ops.push(alloca_op);

        // Compile the expression assigned to the variable
        match self.compile_exp(node_idx) {
            Ok(res_id) => {
                // Store the result of the compiled expression to the variable
                let store_op = Op::new(None, OpData::MemrefStore(operand_id, res_id));
                self.ops.push(store_op);
            }
            Err(e) => return Err(e),
        }
        Ok(())
    }

    fn compile_ret(&mut self, node_idx: usize) -> Result<(), FrontendError> {
        let ret_op_id = match self.compile_exp(node_idx) {
            Err(e) => return Err(e),
            Ok(op_id) => op_id,
        };
        let op_data = OpData::FuncReturn(ret_op_id);
        self.ops.push(Op::new(None, op_data));

        Ok(())
    }

    fn compile_ifelse(&mut self, cond_expr_idx: usize, if_block: &Vec<usize>, else_block: &Option<Vec<usize>>) -> Result<(), FrontendError> {
        todo!()
    }

    fn compile_exp(&mut self, node_idx: usize) -> Result<OperandId, FrontendError> {
        let ast_store = self.ast_store.clone();

        if let ASTNodeKind::Term(_) = &ast_store[node_idx].kind {
            return self.compile_term(node_idx);
        }

        let exp: &Expr = match &ast_store[node_idx].kind {
            ASTNodeKind::Expr(exp) => exp,
            node => panic!("Expected expression in compile_exp, got {:?} instead", node),
        };

        let op_data: OpData = match exp {
            Expr::BinopExp(binop, lhs, rhs) => {
                let (lhs_op, rhs_op) = match (self.compile_exp(*lhs), self.compile_exp(*rhs)) {
                    (Ok(lhs_op), Ok(rhs_op)) => (lhs_op, rhs_op),
                    (Err(err), _) => return Err(err),
                    (_, Err(err)) => return Err(err),
                }; 

                match binop {
                    crate::ast::Binop::Add => OpData::ArithAddI64(lhs_op, rhs_op),
                    crate::ast::Binop::Sub => OpData::ArithSubI64(lhs_op, rhs_op),
                    crate::ast::Binop::Mul => OpData::ArithMulI64(lhs_op, rhs_op),
                    crate::ast::Binop::And => OpData::ArithAndI64(lhs_op, rhs_op),
                    crate::ast::Binop::Or => OpData::ArithOrI64(lhs_op, rhs_op),
                    crate::ast::Binop::Eq => OpData::CmpEqI64(lhs_op, rhs_op), // TODO also support boolean compares
                    crate::ast::Binop::Neq => OpData::CmpNeqI64(lhs_op, rhs_op), // TODO also support boolean compares
                    crate::ast::Binop::Lt => OpData::CmpLtI64(lhs_op, rhs_op),
                    crate::ast::Binop::Lte => OpData::CmpLteI64(lhs_op, rhs_op),
                    crate::ast::Binop::Gt => OpData::CmpGtI64(lhs_op, rhs_op),
                    crate::ast::Binop::Gte => OpData::CmpGteI64(lhs_op, rhs_op),
                }
            }
        };

        let res_id = self.id_gen.gen_id("binop");

        self.ops.push(Op::new(Some(res_id.clone()), op_data));
        Ok(res_id)
    }

    /// Compiles a terminal. Returns the Operand ID that contains the value of the terminal.
    fn compile_term(&mut self, node_idx: usize) -> Result<OperandId, FrontendError> {
        let ast_store = self.ast_store.clone();

        let term: &Term = match &ast_store[node_idx].kind {
            ASTNodeKind::Term(term) => term,
            node => panic!("Expected terminal in compile_term, got {:?} instead", node),
        };

        let op_data: OpData = match term {
            Term::Int(val) => OpData::ArithI64Constant(*val),
            Term::Double(_) => {
                todo!()
            }
            Term::Var(var_id) => {
                let operand_id = match self.ctxt.scope.get(var_id) {
                    None => {
                        return Err(FrontendError {
                            loc: ast_store[node_idx].loc.clone(),
                            msg: format!("Variable {} not found in scope.", var_id),
                        })
                    }
                    Some((metadata, _)) => &metadata.operand_id,
                };
                if !self.ctxt.scope.contains_key(var_id) {
                    return Err(FrontendError {
                        loc: ast_store[node_idx].loc.clone(),
                        msg: format!("Variable {} not found in scope.", var_id),
                    });
                }
                OpData::MemrefLoad(operand_id.clone())
            }
            Term::FuncCall(id, args) => {
                let mut arg_expr_ops: Vec<OperandId> = vec![];
                for arg_id in args {
                    arg_expr_ops.push(self.compile_exp(*arg_id)?);
                }
                OpData::FuncCall(id.to_string(), arg_expr_ops)
            }
        };

        let res_id = self.id_gen.gen_id("term");
        self.ops.push(Op::new(Some(res_id.clone()), op_data));

        Ok(res_id)
    }
}
