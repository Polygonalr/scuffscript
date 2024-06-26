# ScuffScript Compiler

The frontend compiler for ScuffScript, written mainly for educational purposes to practice lexing and parsing via hand-written code.

## Internals of ScuffScript's AST

A ScuffScript program is represented with a list of function declarations (`FDecl` in the AST), each comprising of a list of statements.
- As of now, statements are either of the following:
- Variable declaration
- Variable assignment
- Value return
- All the statements comprises of expressions, which are compiled to be either constant numerical values or binary operations. (Side note: division is not supported at the moment until advanced types are implemented).

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
- **New focus: translation to MLIR (after having finished MLIR dialect code)**
- *Documentation of learning points... maybe?*
