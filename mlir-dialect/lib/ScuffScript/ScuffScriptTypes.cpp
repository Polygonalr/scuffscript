//===- ScuffScriptTypes.cpp - ScuffScript dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ScuffScript/ScuffScriptTypes.h"

#include "ScuffScript/ScuffScriptDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::scuffscript;

#define GET_TYPEDEF_CLASSES
#include "ScuffScript/ScuffScriptOpsTypes.cpp.inc"

void ScuffScriptDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ScuffScript/ScuffScriptOpsTypes.cpp.inc"
      >();
}
