# ScuffScript MLIR dialect

Built from [LLVM project's MLIR standalone dialect template.](https://github.com/llvm/llvm-project/tree/main/mlir/examples/standalone).

## Building - Component Build

I recommend doing a Component Build for ScuffScript's MLIR dialect.

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
ninja
ninja check-scuffscript
```
To build the documentation from the TableGen description of the dialect operations, run
```sh
ninja mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

*For more in-depth steps on how to build everything from scratch, [refer to my gist here.](https://gist.github.com/Polygonalr/5348b6b2d1893eec2fb2faa667b44997)*
