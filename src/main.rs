mod lexer;
mod parser;
mod llvm;
mod ast;

fn main() {
    use crate::lexer::tokenize;
    use crate::parser::Parser;
    {
        let tokens = tokenize("function main () { return 1+2*3; }").unwrap();
        println!("Read tokens {:?}", tokens);
        let mut parser = Parser::from(tokens);
        parser.parse_global().unwrap();
        println!("{}", parser.to_string());
    }
    {
        let tokens = tokenize("function main () { let x = (40+2)*3; return x*2+(3-4)*5+6; }").unwrap();
        println!("Read tokens {:?}", tokens);
        let mut parser = Parser::from(tokens);
        parser.parse_global().unwrap();
        println!("{}", parser.to_string());
    }
}
