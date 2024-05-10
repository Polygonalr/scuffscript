add_mlir_dialect(ScuffScriptOps scuffscript)
add_mlir_doc(ScuffScriptDialect ScuffScriptDialect ScuffScript/ -gen-dialect-doc)
add_mlir_doc(ScuffScriptOps ScuffScriptOps ScuffScript/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS ScuffScriptPasses.td)
mlir_tablegen(ScuffScriptPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRScuffScriptPassesIncGen)
