//===- ScuffScriptTypes.td - ScuffScript dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUFFSCRIPT_TYPES
#define SCUFFSCRIPT_TYPES

include "mlir/IR/AttrTypeBase.td"
include "ScuffScript/ScuffScriptDialect.td"

//===----------------------------------------------------------------------===//
// ScuffScript type definitions
//===----------------------------------------------------------------------===//

class ScuffScript_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ScuffScript_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def ScuffScript_CustomType : ScuffScript_Type<"Custom", "custom"> {
    let summary = "ScuffScript custom type";
    let description = "Custom type in scuffscript dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // SCUFFSCRIPT_TYPES
