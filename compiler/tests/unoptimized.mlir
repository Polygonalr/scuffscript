module {
  llvm.func @main() -> i64 {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.alloca %0 x i64 : (i64) -> !llvm.ptr
    %2 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    %3 = llvm.insertvalue %1, %2[0] : !llvm.struct<(ptr, ptr, i64)> 
    %4 = llvm.insertvalue %1, %3[1] : !llvm.struct<(ptr, ptr, i64)> 
    %5 = llvm.mlir.constant(0 : index) : i64
    %6 = llvm.insertvalue %5, %4[2] : !llvm.struct<(ptr, ptr, i64)> 
    %7 = llvm.mlir.constant(40 : i64) : i64
    %8 = llvm.mlir.constant(2 : i64) : i64
    %9 = llvm.mlir.constant(42 : i64) : i64
    %10 = llvm.mlir.constant(3 : i64) : i64
    %11 = llvm.mul %9, %10 : i64
    %12 = llvm.extractvalue %6[1] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %11, %12 : i64, !llvm.ptr
    %13 = llvm.extractvalue %6[1] : !llvm.struct<(ptr, ptr, i64)> 
    %14 = llvm.load %13 : !llvm.ptr -> i64
    %15 = llvm.mlir.constant(2 : i64) : i64
    %16 = llvm.mul %14, %15 : i64
    %17 = llvm.mlir.constant(3 : i64) : i64
    %18 = llvm.mlir.constant(4 : i64) : i64
    %19 = llvm.mlir.constant(-1 : i64) : i64
    %20 = llvm.mlir.constant(5 : i64) : i64
    %21 = llvm.mul %19, %20 : i64
    %22 = llvm.add %16, %21 : i64
    %23 = llvm.mlir.constant(6 : i64) : i64
    %24 = llvm.add %22, %23 : i64
    llvm.return %24 : i64
  }
}

