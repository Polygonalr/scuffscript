; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define i64 @main() {
  br i1 true, label %1, label %2

1:                                                ; preds = %0
  ret i64 4

2:                                                ; preds = %0
  ret i64 0
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
