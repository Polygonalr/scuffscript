# ScuffScript

**Heavy (but slow) works in progress.**

A TypeScript-like language that compiles to LLVM with the help of MLIR. The main goal is to have a barebones library to interact with the OS's APIs, while supporting a few forms of parallelism, which can include having some GPU APIs to enable GPU computation and a stackful coroutine system that is designed after Golang's goroutines.

This monorepo contains two projects:
- **The ScuffScript compiler** which is written in Rust. It deals with the lexing, parsing and translation of ScuffScript code into its MLIR dialect.
- **The ScuffScript MLIR dialect** which is written in C++, ~~and deals with all the heavy lifting of defining valid operations for ScuffScript's MLIR dialect and lowering them to LLVM~~.
  - Currently unmaintained and unused as I have no confidence in developing custom dialects.

