// LLVM types - for now only support 64 bits
enum LLType {
    Void,
    I64,
    Pointer(Box<LLType>),
    Function(Vec<LLType>, Box<LLType>), // arg types, return type
}
