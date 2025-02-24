use std::rc::Rc;

use crate::ast::Type;

pub type OperandId = Rc<str>;
pub type FuncId = String;

pub enum MlirType {
    I64,
    F64,
    Bool,
    Void,
    Memref(Box<MlirType>),
}

impl From<&Type> for MlirType {
    fn from(t: &Type) -> MlirType {
        match t {
            Type::Int => MlirType::I64,
            Type::Double => MlirType::F64,
            Type::Bool => MlirType::Bool,
            Type::Void => MlirType::Void,
            _ => unimplemented!("Strings are special cases!"),
        }
    }
}

impl ToString for MlirType {
    fn to_string(&self) -> String {
        use MlirType::*;
        match self {
            I64 => "i64".to_string(),
            F64 => "f64".to_string(),
            Bool => "i1".to_string(),
            Void => "void".to_string(),
            Memref(inner) => format!("memref<{}>", inner.to_string()),
        }
    }
}

pub enum CmpPredicate {
    Equal = 0,
    NotEqual = 1,
    LessThan = 2,
    LessThanOrEqual = 3,
    GreaterThan = 4,
    GreatherThanOrEqual = 5,
}

pub enum OpData {
    ArithI64Constant(i64),
    ArithAddI64(OperandId, OperandId),
    ArithSubI64(OperandId, OperandId),
    ArithMulI64(OperandId, OperandId),
    ArithAndI64(OperandId, OperandId),
    ArithOrI64(OperandId, OperandId),
    CmpEqI64(OperandId, OperandId),
    CmpNeqI64(OperandId, OperandId),
    CmpLtI64(OperandId, OperandId),
    CmpLteI64(OperandId, OperandId),
    CmpGtI64(OperandId, OperandId),
    CmpGteI64(OperandId, OperandId),
    MemrefAlloca(MlirType),
    MemrefLoad(OperandId),
    MemrefStore(OperandId, OperandId), // dest, src
    FuncCall(FuncId, Vec<OperandId>),  // func to call, operands to pass as args
    FuncReturn(OperandId),
}

pub struct Op {
    result_id: Option<OperandId>,
    data: OpData,
}

impl Op {
    pub fn new(result_id: Option<OperandId>, data: OpData) -> Self {
        Self { result_id, data }
    }

    pub fn to_ir(&self) -> String {
        use crate::mlir::ast::OpData::*;

        let mut res = self
            .result_id
            .as_ref()
            .map_or("".to_string(), |id| format!("%{} = ", id));

        res.push_str(&match &self.data {
            ArithI64Constant(literal) => format!("arith.constant {} : i64", literal),
            ArithAddI64(op1, op2) => format!("arith.addi %{}, %{} : i64", op1, op2),
            ArithSubI64(op1, op2) => format!("arith.subi %{}, %{} : i64", op1, op2),
            ArithMulI64(op1, op2) => format!("arith.muli %{}, %{} : i64", op1, op2),
            ArithAndI64(op1, op2) => format!("arith.andi %{}, %{} : i64", op1, op2),
            ArithOrI64(op1, op2) => format!("arith.ori %{}, %{} : i64", op1, op2),
            CmpEqI64(op1, op2) => format!("arith.cmpi eq %{}, %{} : i64", op1, op2),
            CmpNeqI64(op1, op2) => format!("arith.cmpi ne %{}, %{} : i64", op1, op2),
            CmpLtI64(op1, op2) => format!("arith.cmpi slt %{}, %{} : i64", op1, op2),
            CmpLteI64(op1, op2) => format!("arith.cmpi sle %{}, %{} : i64", op1, op2),
            CmpGtI64(op1, op2) => format!("arith.cmpi sgt %{}, %{} : i64", op1, op2),
            CmpGteI64(op1, op2) => format!("arith.cmpi sge %{}, %{} : i64", op1, op2),
            MemrefAlloca(mlir_type) => {
                format!("memref.alloca() : memref<{}>", mlir_type.to_string())
            }
            MemrefLoad(src_op) => format!("memref.load %{}[] : memref<i64>", src_op), // TODO remove i64 hardcode
            MemrefStore(dest_op, val_op) => {
                format!("memref.store %{}, %{}[] : memref<i64>", val_op, dest_op)
            } // TODO remove i64 hardcode
            FuncCall(func_id, args) => {
                let args_string = args.iter().map(|s| format!("%{}", s)).collect::<Vec<String>>().join(", ");
                let mut types_string = "i64, ".repeat(args.len());
                if !types_string.is_empty() {
                    types_string.pop();
                    types_string.pop();
                }
                format!(
                    "func.call @{}({}) : ({}) -> {}",
                    func_id, args_string, types_string, "i64"
                )
            } // TODO remove i64 hardcode
            FuncReturn(op) => format!("func.return %{} : i64", op), // TODO remove i64 hardcode
        });

        res
    }
}

pub struct Func {
    identifier: FuncId,
    params: Vec<(OperandId, MlirType)>,
    operations: Vec<Op>,
}

impl Func {
    pub fn new(identifier: FuncId) -> Self {
        Self {
            identifier,
            params: Vec::new(),
            operations: Vec::new(),
        }
    }

    pub fn append_param(&mut self, param: (OperandId, MlirType)) {
        self.params.push(param);
    }

    pub fn append_ops(&mut self, ops: Vec<Op>) {
        self.operations.extend(ops);
    }

    pub fn to_ir(&self) -> String {
        let mut output = format!("func.func @{}(\n", self.identifier);

        for (i, (param_id, param_type)) in self.params.iter().enumerate() {
            output.push_str(&format!("    %{}: {}", param_id, param_type.to_string()));
            if i != self.params.len() - 1 {
                output.push_str(",\n");
            }
        }
        output.push_str(") -> i64\n{\n"); // TODO For now everything is int
        for op in &self.operations {
            output.push_str(&format!("    {}\n", op.to_ir()));
        }
        output.push_str("}\n");

        output
    }
}

pub struct MlirProg {
    functions: Vec<Func>, // TODO add global declarations
}

impl MlirProg {
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
        }
    }

    pub fn add_func(&mut self, func: Func) {
        self.functions.push(func);
    }

    pub fn to_ir(&self) -> String {
        let mut output: String = "".to_string();
        for func in &self.functions {
            output.push_str(&func.to_ir());
        }
        output
    }
}
