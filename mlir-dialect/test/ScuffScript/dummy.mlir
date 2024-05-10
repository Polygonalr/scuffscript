// RUN: scuffscript-opt %s | scuffscript-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = scuffscript.foo %{{.*}} : i32
        %res = scuffscript.foo %0 : i32
        return
    }

    // CHECK-LABEL: func @scuffscript_types(%arg0: !scuffscript.custom<"10">)
    func.func @scuffscript_types(%arg0: !scuffscript.custom<"10">) {
        return
    }
}
