include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_scuffscript`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_scuffscript.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(ScuffScriptPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT ScuffScriptPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_scuffscript"
  TD_FILE dialects/ScuffScriptOps.td
  SOURCES
    dialects/scuffscript.py
  DIALECT_NAME scuffscript)

declare_mlir_python_extension(ScuffScriptPythonSources.Extension
  MODULE_NAME _scuffscriptDialects
  ADD_TO_PARENT ScuffScriptPythonSources
  SOURCES
    ScuffScriptExtension.cpp
  EMBED_CAPI_LINK_LIBS
    ScuffScriptCAPI
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(ScuffScriptPythonCAPI
  INSTALL_COMPONENT ScuffScriptPythonModules
  INSTALL_DESTINATION python_packages/scuffscript/mlir_scuffscript/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/scuffscript/mlir_scuffscript/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    ScuffScriptPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(ScuffScriptPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/scuffscript/mlir_scuffscript"
  INSTALL_PREFIX "python_packages/scuffscript/mlir_scuffscript"
  DECLARED_SOURCES
    ScuffScriptPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    ScuffScriptPythonCAPI
  )
