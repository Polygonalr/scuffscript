//===- ScuffScriptPasses.h - ScuffScript passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef SCUFFSCRIPT_SCUFFSCRIPTPASSES_H
#define SCUFFSCRIPT_SCUFFSCRIPTPASSES_H

#include "ScuffScript/ScuffScriptDialect.h"
#include "ScuffScript/ScuffScriptOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace scuffscript {
#define GEN_PASS_DECL
#include "ScuffScript/ScuffScriptPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "ScuffScript/ScuffScriptPasses.h.inc"
} // namespace scuffscript
} // namespace mlir

#endif
