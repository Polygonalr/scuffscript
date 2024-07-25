; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define i64 @add(i64 %0, i64 %1) {
  %3 = alloca i64, i64 1, align 8
  %4 = insertvalue { ptr, ptr, i64 } undef, ptr %3, 0
  %5 = insertvalue { ptr, ptr, i64 } %4, ptr %3, 1
  %6 = insertvalue { ptr, ptr, i64 } %5, i64 0, 2
  %7 = extractvalue { ptr, ptr, i64 } %6, 1
  store i64 %0, ptr %7, align 4
  %8 = alloca i64, i64 1, align 8
  %9 = insertvalue { ptr, ptr, i64 } undef, ptr %8, 0
  %10 = insertvalue { ptr, ptr, i64 } %9, ptr %8, 1
  %11 = insertvalue { ptr, ptr, i64 } %10, i64 0, 2
  %12 = extractvalue { ptr, ptr, i64 } %11, 1
  store i64 %1, ptr %12, align 4
  %13 = extractvalue { ptr, ptr, i64 } %6, 1
  %14 = load i64, ptr %13, align 4
  %15 = extractvalue { ptr, ptr, i64 } %11, 1
  %16 = load i64, ptr %15, align 4
  %17 = add i64 %14, %16
  ret i64 %17
}

define i64 @main() {
  %1 = call i64 @add(i64 2, i64 3)
  ret i64 %1
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
