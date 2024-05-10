llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(SCUFFSCRIPT_TEST_DEPENDS
        FileCheck count not
        scuffscript-capi-test
        scuffscript-opt
        scuffscript-translate
        ScuffScriptPlugin
        )
if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND SCUFFSCRIPT_TEST_DEPENDS ScuffScriptPythonModules)
endif()

add_lit_testsuite(check-scuffscript "Running the scuffscript regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${SCUFFSCRIPT_TEST_DEPENDS}
        )
set_target_properties(check-scuffscript PROPERTIES FOLDER "Tests")

add_lit_testsuites(SCUFFSCRIPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${SCUFFSCRIPT_TEST_DEPENDS})

add_subdirectory(CAPI)
