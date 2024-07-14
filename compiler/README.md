# ScuffScript Compiler

The frontend compiler for ScuffScript, written mainly for educational purposes to practice lexing and parsing via hand-written code.

## Internals of ScuffScript's AST

A ScuffScript program is represented with a list of function declarations (`FDecl` in the AST), each comprising of a list of statements.
- As of now, statements are either of the following:
- Variable declaration
- Variable assignment
- Value return
- All the statements comprises of expressions, which are compiled to be either constant numerical values or binary operations. (Side note: division is not supported at the moment until advanced types are implemented).

## Translation to MLIR

- No custom dialect, relies on `arith`, `func`, `llvm` dialects for now. `arith` and `func` will eventually be lowered to the `llvm` dialect in the later parts of the compilation pipeline.
  - Doing so to understand MLIR more first before delving into developing a custom dialect for ScuffScript.
- There are plans on integrating other dialects like `memref`, `gpu`, `omp`.

## Some code naming glossary
- `ast`: Abstract Syntax Tree
- `stmt`: Statement
- `exp`: Expression
- `term`: Terminal
- `fdecl`: Function Declaration
- `id`: Identifier (not index, which is denoted with `idx`)

## To-dos
- Implement the parser for only 1 (main) function.
- Implement function calls, and support for multiple functions.
- Implement a simple standard library with a runtime helper written in C/C++.
- ~~(Vague) Start on implementing LLVM AST and code translation.~~ 
- ~~(Vague) Implement LLVM AST program printer~~
- ~~Update main.rs to pass the generated LLVM program string into clang.~~
- **New focus: translation to MLIR (scuffed attempt with using the stock dialects).**
- Basic type checking.
- *Documentation of learning points... maybe?*
