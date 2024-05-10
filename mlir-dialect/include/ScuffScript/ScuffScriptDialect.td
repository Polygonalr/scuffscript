//===- ScuffScriptDialect.td - ScuffScript dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUFFSCRIPT_DIALECT
#define SCUFFSCRIPT_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ScuffScript dialect definition.
//===----------------------------------------------------------------------===//

def ScuffScript_Dialect : Dialect {
    let name = "scuffscript";
    let summary = "A scuffscript out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::scuffscript";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base scuffscript operation definition.
//===----------------------------------------------------------------------===//

class ScuffScript_Op<string mnemonic, list<Trait> traits = []> :
        Op<ScuffScript_Dialect, mnemonic, traits>;

#endif // SCUFFSCRIPT_DIALECT
