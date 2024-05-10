add_llvm_library(ScuffScriptPlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        scuffscript-plugin.cpp

        DEPENDS
        MLIRScuffScript
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIRScuffScript
        )
target_include_directories(
        ScuffScriptPlugin
        PRIVATE
        "${SCUFFSCRIPT_BINARY_DIR}/include"
)
