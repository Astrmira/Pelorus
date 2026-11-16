//! 编译Linux原生io_uring窄接口，其他平台使用纯Rust存储路径。

use std::{env, path::PathBuf, process::Command};

fn run(command: &mut Command) {
    let status = command.status().expect("启动C工具链失败");
    assert!(status.success(), "io_uring桥接编译或归档失败");
}

fn main() {
    println!("cargo:rerun-if-changed=src/storage/uring_bridge.c");
    println!("cargo:rerun-if-env-changed=CC");
    println!("cargo:rerun-if-env-changed=AR");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("缺少构建输出目录"));
    let object = out.join("uring_bridge.o");
    let archive = out.join("libpelorus_uring.a");
    run(Command::new(env::var("CC").unwrap_or_else(|_| "cc".into()))
        .args([
            "-O3",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fPIC",
            "-c",
            "src/storage/uring_bridge.c",
            "-o",
        ])
        .arg(&object));
    run(Command::new(env::var("AR").unwrap_or_else(|_| "ar".into()))
        .arg("crs")
        .arg(&archive)
        .arg(&object));
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=pelorus_uring");
}
