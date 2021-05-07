use std::{fs, path::Path};

use shaderc::{Compiler, ShaderKind};
use std::str;
fn main() {
    // TODO: support dynamic shader compilation
    println!("cargo:rerun-if-changed=src/tutorials/shaders/");

    let mut compiler = Compiler::new().expect("Unable to instantiate compiler");

    let mut file = Path::new("./src/tutorials/shaders/shader.vert").to_path_buf();
    let source_text = fs::read(&file).expect("Unable to read string");
    let source_text = str::from_utf8(&source_text).expect("Unable to parse string");
    let artifact = compiler
        .compile_into_spirv(
            source_text,
            ShaderKind::Vertex,
            file.file_name().unwrap().to_str().unwrap(),
            "main",
            None,
        )
        .expect("Unable to compile file");
    file.set_extension("vert.spv");
    fs::write(&file, &artifact.as_binary_u8()).expect("Unable to write spirv to file");

    let mut file = Path::new("./src/tutorials/shaders/shader.frag").to_path_buf();
    let source_text = fs::read(&file).expect("Unable to read string");
    let source_text = str::from_utf8(&source_text).expect("Unable to parse string");
    let artifact = compiler
        .compile_into_spirv(
            source_text,
            ShaderKind::Fragment,
            file.file_name().unwrap().to_str().unwrap(),
            "main",
            None,
        )
        .expect("Unable to compile file");
    file.set_extension("frag.spv");
    fs::write(&file, &artifact.as_binary_u8()).expect("Unable to write spirv to file");
}
