func.func @main() -> i64 {
    %x0 = memref.alloca() : memref<i64>
    %term1 = arith.constant 40 : i64
    %term2 = arith.constant 2 : i64
    %binop3 = arith.addi %term1, %term2 : i64
    %term4 = arith.constant 3 : i64
    %binop5 = arith.muli %binop3, %term4 : i64
    memref.store %binop5, %x0[] : memref<i64>
    %term6 = memref.load %x0[] : memref<i64>
    %term7 = arith.constant 2 : i64
    %binop8 = arith.muli %term6, %term7 : i64
    %term9 = arith.constant 3 : i64
    %term10 = arith.constant 4 : i64
    %binop11 = arith.subi %term9, %term10 : i64
    %term12 = arith.constant 5 : i64
    %binop13 = arith.muli %binop11, %term12 : i64
    %binop14 = arith.addi %binop8, %binop13 : i64
    %term15 = arith.constant 6 : i64
    %binop16 = arith.addi %binop14, %term15 : i64
    func.return %binop16 : i64
}