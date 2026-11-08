//! Platform probes: machine description and (on Apple Silicon) SoC energy.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Machine {
    pub os: String,
    pub arch: String,
    pub chip: String,
    pub memory_bytes: Option<u64>,
}

/// Energy spent between [`PowerMeter::begin`] and [`PowerMeter::end`], millijoules.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Energy {
    pub cpu_mj: f64,
    pub gpu_mj: f64,
    pub ane_mj: f64,
    pub dram_mj: f64,
}

impl Energy {
    pub fn total_mj(&self) -> f64 {
        self.cpu_mj + self.gpu_mj + self.ane_mj + self.dram_mj
    }
}

#[cfg(target_os = "macos")]
mod mac {
    use super::Energy;

    #[repr(C)]
    #[derive(Default)]
    pub struct KldEnergy {
        cpu_mj: f64,
        gpu_mj: f64,
        ane_mj: f64,
        dram_mj: f64,
    }

    extern "C" {
        fn kld_power_init() -> i32;
        fn kld_power_begin() -> i32;
        fn kld_power_end(out: *mut KldEnergy) -> i32;
        pub fn kld_phys_footprint() -> u64;
        fn kld_chip_name(buf: *mut u8, len: i32) -> i32;
        pub fn kld_total_memory() -> u64;
    }

    pub fn init() -> bool {
        unsafe { kld_power_init() == 0 }
    }
    pub fn begin() -> bool {
        unsafe { kld_power_begin() == 0 }
    }
    pub fn end() -> Option<Energy> {
        let mut e = KldEnergy::default();
        if unsafe { kld_power_end(&mut e) } != 0 {
            return None;
        }
        Some(Energy { cpu_mj: e.cpu_mj, gpu_mj: e.gpu_mj, ane_mj: e.ane_mj, dram_mj: e.dram_mj })
    }
    pub fn chip() -> String {
        let mut buf = [0u8; 128];
        let n = unsafe { kld_chip_name(buf.as_mut_ptr(), buf.len() as i32) };
        String::from_utf8_lossy(&buf[..n.max(0) as usize]).into_owned()
    }
}

/// System-wide SoC energy meter. A no-op everywhere except Apple Silicon.
pub struct PowerMeter {
    enabled: bool,
}

impl PowerMeter {
    pub fn new() -> Self {
        #[cfg(target_os = "macos")]
        {
            Self { enabled: mac::init() }
        }
        #[cfg(not(target_os = "macos"))]
        {
            Self { enabled: false }
        }
    }

    pub fn available(&self) -> bool {
        self.enabled
    }

    pub fn begin(&mut self) {
        #[cfg(target_os = "macos")]
        if self.enabled {
            self.enabled = mac::begin();
        }
    }

    pub fn end(&mut self) -> Option<Energy> {
        #[cfg(target_os = "macos")]
        if self.enabled {
            return mac::end();
        }
        None
    }

    /// Average idle power in watts over `secs`, used as a baseline so that
    /// background activity is not charged to the model.
    pub fn idle_watts(&mut self, secs: f64) -> Option<f64> {
        if !self.enabled {
            return None;
        }
        self.begin();
        std::thread::sleep(std::time::Duration::from_secs_f64(secs));
        self.end().map(|e| e.total_mj() / 1000.0 / secs)
    }
}

/// Physical memory footprint of this process (macOS), bytes.
pub fn process_footprint() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        Some(unsafe { mac::kld_phys_footprint() })
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// Marketing name of the CPU on Windows, e.g. "AMD Ryzen 5 7640HS".
fn windows_cpu_name() -> Option<String> {
    if !cfg!(windows) {
        return None;
    }
    let out = std::process::Command::new("reg")
        .args(["query", r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0", "/v", "ProcessorNameString"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().find(|l| l.contains("ProcessorNameString"))?;
    Some(line.split("REG_SZ").nth(1)?.trim().to_string()).filter(|s| !s.is_empty())
}

/// Collapse `a/b/../c` into `a/c` without touching the filesystem.
pub fn clean_path(p: &std::path::Path) -> std::path::PathBuf {
    use std::path::Component;
    let mut out = std::path::PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir if matches!(out.components().next_back(), Some(Component::Normal(_))) => {
                out.pop();
            }
            other => out.push(other),
        }
    }
    out
}

pub fn machine() -> Machine {
    let os = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();

    #[cfg(target_os = "macos")]
    {
        return Machine { os, arch, chip: mac::chip(), memory_bytes: Some(unsafe { mac::kld_total_memory() }) };
    }

    #[allow(unreachable_code)]
    {
        let chip = windows_cpu_name()
            .or_else(|| std::env::var("PROCESSOR_IDENTIFIER").ok())
            .or_else(|| {
                std::fs::read_to_string("/proc/cpuinfo").ok().and_then(|s| {
                    s.lines()
                        .find(|l| l.starts_with("model name"))
                        .and_then(|l| l.split(':').nth(1))
                        .map(|s| s.trim().to_string())
                })
            })
            .unwrap_or_else(|| "unknown CPU".into());
        Machine { os, arch, chip, memory_bytes: None }
    }
}
