// RUN: mlir-opt %s --load-dialect-plugin=%scuffscript_libs/ScuffScriptPlugin%shlibext --pass-pipeline="builtin.module(scuffscript-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @scuffscript_types(%arg0: !scuffscript.custom<"10">)
  func.func @scuffscript_types(%arg0: !scuffscript.custom<"10">) {
    return
  }
}
