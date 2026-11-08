fn main() {
    println!("cargo:rerun-if-changed=objc/kld_mac.m");
    println!("cargo:rerun-if-changed=workers/mlx_worker.py");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        cc::Build::new()
            .file("objc/kld_mac.m")
            .flag("-fobjc-arc")
            .flag("-Wno-unused-parameter")
            .compile("kld_mac");
        for fw in ["Foundation", "CoreFoundation", "IOKit"] {
            println!("cargo:rustc-link-lib=framework={fw}");
        }
        // Private SoC energy-counter library, present on every Apple Silicon Mac.
        println!("cargo:rustc-link-lib=dylib=IOReport");
    }
}
