// RUN: mlir-opt %s --load-pass-plugin=%scuffscript_libs/ScuffScriptPlugin%shlibext --pass-pipeline="builtin.module(scuffscript-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @abar()
  func.func @abar() {
    return
  }
}
