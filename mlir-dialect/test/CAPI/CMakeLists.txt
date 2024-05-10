# Here we create a single aggregate shared library with the parts of the CAPI
# that we want to bundle together. Then we link a simple C executable
# against it to demonstrate that it does have the fully self contained
# core MLIR library and our own scuffscript dialect.
add_mlir_aggregate(ScuffScriptCAPITestLib
  SHARED
  EMBED_LIBS
  MLIRCAPIIR
  # TODO: Remove this in favor of showing fine grained dialect registration
  # (once available).
  MLIRCAPIRegisterEverything
  ScuffScriptCAPI
)

add_llvm_executable(scuffscript-capi-test
  scuffscript-capi-test.c
)
llvm_update_compile_flags(scuffscript-capi-test)
target_link_libraries(scuffscript-capi-test
  PRIVATE ScuffScriptCAPITestLib)
