# RUN: %python %s | FileCheck %s

from mlir_scuffscript.ir import *
from mlir_scuffscript.dialects import builtin as builtin_d, scuffscript as scuffscript_d

with Context():
    scuffscript_d.register_dialect()
    module = Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = scuffscript.foo %0 : i32
    """
    )
    # CHECK: %[[C:.*]] = arith.constant 2 : i32
    # CHECK: scuffscript.foo %[[C]] : i32
    print(str(module))
