//===- ScuffScriptDialect.cpp - ScuffScript dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ScuffScript/ScuffScriptDialect.h"
#include "ScuffScript/ScuffScriptOps.h"
#include "ScuffScript/ScuffScriptTypes.h"

using namespace mlir;
using namespace mlir::scuffscript;

#include "ScuffScript/ScuffScriptOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// ScuffScript dialect.
//===----------------------------------------------------------------------===//

void ScuffScriptDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ScuffScript/ScuffScriptOps.cpp.inc"
      >();
  registerTypes();
}
