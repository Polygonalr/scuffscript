; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define i64 @main() {
  %1 = alloca i64, i64 1, align 8
  %2 = insertvalue { ptr, ptr, i64 } undef, ptr %1, 0
  %3 = insertvalue { ptr, ptr, i64 } %2, ptr %1, 1
  %4 = insertvalue { ptr, ptr, i64 } %3, i64 0, 2
  %5 = alloca i64, i64 1, align 8
  %6 = insertvalue { ptr, ptr, i64 } undef, ptr %5, 0
  %7 = insertvalue { ptr, ptr, i64 } %6, ptr %5, 1
  %8 = insertvalue { ptr, ptr, i64 } %7, i64 0, 2
  %9 = extractvalue { ptr, ptr, i64 } %4, 1
  store i64 666, ptr %9, align 4
  %10 = extractvalue { ptr, ptr, i64 } %4, 1
  %11 = load i64, ptr %10, align 4
  %12 = sub i64 %11, 596
  %13 = extractvalue { ptr, ptr, i64 } %8, 1
  store i64 %12, ptr %13, align 4
  %14 = extractvalue { ptr, ptr, i64 } %8, 1
  %15 = load i64, ptr %14, align 4
  %16 = add i64 %15, 10
  ret i64 %16
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
