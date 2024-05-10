add_mlir_dialect_library(MLIRScuffScript
        ScuffScriptTypes.cpp
        ScuffScriptDialect.cpp
        ScuffScriptOps.cpp
        ScuffScriptPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ScuffScript

        DEPENDS
        MLIRScuffScriptOpsIncGen
        MLIRScuffScriptPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
