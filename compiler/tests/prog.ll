; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define i64 @main() {
  %1 = alloca i64, i64 1, align 8
  %2 = insertvalue { ptr, ptr, i64 } undef, ptr %1, 0
  %3 = insertvalue { ptr, ptr, i64 } %2, ptr %1, 1
  %4 = insertvalue { ptr, ptr, i64 } %3, i64 0, 2
  %5 = extractvalue { ptr, ptr, i64 } %4, 1
  store i64 126, ptr %5, align 4
  %6 = extractvalue { ptr, ptr, i64 } %4, 1
  %7 = load i64, ptr %6, align 4
  %8 = mul i64 %7, 2
  %9 = add i64 %8, -5
  %10 = add i64 %9, 6
  ret i64 %10
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
