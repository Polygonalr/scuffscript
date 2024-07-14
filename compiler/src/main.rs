mod ast;
mod frontend;
mod lexer;
mod mlir;
mod parser;

use clap::Parser;
use std::fs;
use std::io::{Read, Write};
use std::process::Command;
use which::which;

use crate::frontend::Frontend;
use crate::lexer::tokenize;
use crate::mlir::ast::MlirProg;
use crate::parser::Parser as SSParser;

#[derive(clap::Parser, Debug)]
struct Cli {
    /// Path of the ScuffScript source file
    input_path: String,

    /// Path of the output program
    #[arg(short, long)]
    output: Option<String>,

    #[arg(long)]
    /// Compile to LLVM source file instead of executable
    ll: bool,
}

fn check_mlir_components() {
    match which("mlir-translate") {
        Ok(_) => (),
        Err(_) => {
            println!("mlir-translate is not found in PATH.");
            println!(
                "Please make sure that the MLIR tools are installed and available in your PATH."
            );
            std::process::exit(1);
        }
    }

    match which("mlir-opt") {
        Ok(_) => (),
        Err(_) => {
            println!("mlir-opt is not found in PATH.");
            println!(
                "Please make sure that the MLIR tools are installed and available in your PATH."
            );
            std::process::exit(1);
        }
    }

    match which("clang") {
        Ok(_) => (),
        Err(_) => {
            println!("clang is not found in PATH.");
            println!("Please make sure that the clang is installed and available in your PATH.");
            std::process::exit(1);
        }
    }
}

fn main() {
    let cli = Cli::parse();
    let input = fs::read_to_string(cli.input_path).expect("Failed to read input file");

    check_mlir_components();

    let tokens = tokenize(&input).unwrap();
    let mut parser = SSParser::from(tokens);
    parser.parse().unwrap();
    let mut frontend = Frontend::new(parser.ast_store, parser.func_decls);
    let mlir_prog: MlirProg = frontend.compile_to_mlir().unwrap();

    // Compile the IR
    let ir_input = mlir_prog.to_ir().into_bytes();
    let mut cmd = Command::new("mlir-opt")
        .arg("--convert-arith-to-llvm")
        .arg("--convert-func-to-llvm")
        .arg("--finalize-memref-to-llvm")
        .arg("-")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("Failed to spawn mlir-opt");
    let mut stdin = cmd.stdin.take().unwrap();
    let mut stdout = cmd.stdout.take().unwrap();
    Write::write_all(&mut stdin, &ir_input).expect("Failed to write to mlir-opt");
    stdin.flush().unwrap();
    drop(stdin);

    let mut mlir_opt_output = Vec::new();
    stdout.read_to_end(&mut mlir_opt_output).unwrap();

    let mut cmd = Command::new("mlir-translate")
        .arg("--mlir-to-llvmir")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("Failed to spawn mlir-translate");
    let mut stdin = cmd.stdin.take().unwrap();
    let mut stdout = cmd.stdout.take().unwrap();
    Write::write_all(&mut stdin, &mlir_opt_output).expect("Failed to write to mlir-translate");
    stdin.flush().unwrap();
    drop(stdin);

    let mut llvm_prog = Vec::new();
    stdout.read_to_end(&mut llvm_prog).unwrap();

    if cli.ll {
        if let Some(output_path) = cli.output {
            let mut output_file =
                fs::File::create(output_path).expect("Failed to create output file");
            output_file
                .write_all(&llvm_prog)
                .expect("Failed to write to output file");
        } else {
            println!("LLVM IR: {}", String::from_utf8_lossy(&llvm_prog));
        }
    } else if let Some(output_path) = cli.output {
        let mut cmd = Command::new("clang")
            .arg("-O3")
            .arg("-x")
            .arg("ir")
            .arg("-")
            .arg("-o")
            .arg(output_path)
            .stdin(std::process::Stdio::piped())
            .spawn()
            .expect("Failed to spawn clang");
        let mut stdin = cmd.stdin.take().unwrap();
        Write::write_all(&mut stdin, &llvm_prog).expect("Failed to write to clang");
        stdin.flush().unwrap();
        drop(stdin);
    } else {
        println!("Error: missing output path");
    }
}
