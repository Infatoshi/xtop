use crate::app::{App, CoreData, ProcessInfo, CORE_HISTORY_SIZE, HISTORY_SIZE};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use sysinfo::{Components, CpuRefreshKind, Networks, ProcessRefreshKind, RefreshKind, System};

#[cfg(target_os = "linux")]
use crate::app::GpuData;
#[cfg(target_os = "linux")]
use nvml_wrapper::enums::device::UsedGpuMemory;
#[cfg(target_os = "linux")]
use nvml_wrapper::Nvml;

pub struct Collectors {
    sys: System,
    networks: Networks,
    components: Components,
    #[cfg(target_os = "linux")]
    nvml: Option<Nvml>,
    #[cfg(target_os = "linux")]
    driver_version: String,
    #[cfg(target_os = "linux")]
    cuda_version: String,
    prev_net: Option<HashMap<String, (u64, u64)>>,
    prev_disk: Option<HashMap<String, (u64, u64)>>,
}

impl Collectors {
    pub fn new() -> Self {
        let sys = System::new_with_specifics(
            RefreshKind::everything().with_cpu(CpuRefreshKind::everything()),
        );
        let networks = Networks::new_with_refreshed_list();
        let components = Components::new_with_refreshed_list();

        #[cfg(target_os = "linux")]
        let nvml = Nvml::init().ok();
        #[cfg(target_os = "linux")]
        let (driver_version, cuda_version) = if let Some(ref nvml) = nvml {
            let driver = nvml.sys_driver_version().unwrap_or_default();
            let cuda_major = nvml.sys_cuda_driver_version().unwrap_or(0) / 1000;
            let cuda_minor = (nvml.sys_cuda_driver_version().unwrap_or(0) % 1000) / 10;
            (driver, format!("{}.{}", cuda_major, cuda_minor))
        } else {
            (String::new(), String::new())
        };

        Self {
            sys,
            networks,
            components,
            #[cfg(target_os = "linux")]
            nvml,
            #[cfg(target_os = "linux")]
            driver_version,
            #[cfg(target_os = "linux")]
            cuda_version,
            prev_net: None,
            prev_disk: None,
        }
    }

    pub fn collect(&mut self, app: &mut App, interval_secs: f64) {
        self.sys.refresh_all();
        self.networks.refresh(true);
        self.components.refresh(true);

        self.collect_cpu(app);
        self.collect_memory(app);
        self.collect_gpu(app);
        self.collect_network(app, interval_secs);
        self.collect_disk(app, interval_secs);
        self.collect_battery(app);
        self.collect_processes(app);
    }

    fn collect_cpu(&mut self, app: &mut App) {
        let cpu_info = self.sys.cpus();
        if cpu_info.is_empty() {
            return;
        }

        app.cpu.model = self.sys.cpus().first().map(|c| c.brand().to_string()).unwrap_or_default();
        app.cpu.core_count = cpu_info.len();
        app.cpu.usage_percent = self.sys.global_cpu_usage();

        // Get max frequency across cores
        app.cpu.frequency_mhz = cpu_info.iter().map(|c| c.frequency()).max().unwrap_or(0);

        // Get CPU temperature from components
        app.cpu.temperature_c = self.components.iter()
            .find(|c| {
                let label = c.label().to_lowercase();
                label.contains("tctl") || label.contains("cpu") || label.contains("core")
            })
            .and_then(|c| c.temperature());

        // Update overall history
        if app.cpu.history.len() >= HISTORY_SIZE {
            app.cpu.history.pop_front();
        }
        app.cpu.history.push_back(app.cpu.usage_percent);

        // Update per-core data
        while app.cpu.cores.len() < cpu_info.len() {
            app.cpu.cores.push(CoreData::default());
        }
        app.cpu.cores.truncate(cpu_info.len());

        for (i, cpu) in cpu_info.iter().enumerate() {
            let core = &mut app.cpu.cores[i];
            core.usage_percent = cpu.cpu_usage();

            if core.history.len() >= CORE_HISTORY_SIZE {
                core.history.pop_front();
            }
            core.history.push_back(core.usage_percent);
        }
    }

    fn collect_memory(&self, app: &mut App) {
        app.memory.total_bytes = self.sys.total_memory();
        app.memory.used_bytes = self.sys.used_memory();
        app.memory.swap_total_bytes = self.sys.total_swap();
        app.memory.swap_used_bytes = self.sys.used_swap();

        let usage_percent = if app.memory.total_bytes > 0 {
            (app.memory.used_bytes as f64 / app.memory.total_bytes as f64 * 100.0) as f32
        } else {
            0.0
        };
        if app.memory.history.len() >= HISTORY_SIZE {
            app.memory.history.pop_front();
        }
        app.memory.history.push_back(usage_percent);
    }

    #[cfg(target_os = "linux")]
    fn collect_gpu(&mut self, app: &mut App) {
        let Some(ref nvml) = self.nvml else {
            app.gpus.clear();
            return;
        };

        let device_count = nvml.device_count().unwrap_or(0);

        while app.gpus.len() < device_count as usize {
            app.gpus.push(GpuData::default());
        }
        app.gpus.truncate(device_count as usize);

        for i in 0..device_count {
            let Ok(device) = nvml.device_by_index(i) else {
                continue;
            };

            let gpu = &mut app.gpus[i as usize];
            gpu.index = i;
            gpu.name = device.name().unwrap_or_else(|_| "Unknown".to_string());
            gpu.driver_version = self.driver_version.clone();
            gpu.cuda_version = self.cuda_version.clone();

            gpu.utilization_percent = device
                .utilization_rates()
                .map(|u| u.gpu)
                .unwrap_or(0);

            gpu.temperature_c = device
                .temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
                .unwrap_or(0);

            if let Ok(mem) = device.memory_info() {
                gpu.memory_used_bytes = mem.used;
                gpu.memory_total_bytes = mem.total;
            }

            gpu.power_usage_w = device.power_usage().unwrap_or(0) / 1000;
            gpu.power_limit_w = device
                .enforced_power_limit()
                .or_else(|_| device.power_management_limit())
                .unwrap_or(0) / 1000;

            gpu.fan_speed_percent = device.fan_speed(0).unwrap_or(0);

            if let Ok(cuda_compute) = device.cuda_compute_capability() {
                gpu.compute_major = cuda_compute.major as u32;
                gpu.compute_minor = cuda_compute.minor as u32;
                gpu.architecture = match cuda_compute.major {
                    9 => "Blackwell".to_string(),
                    8 => if cuda_compute.minor >= 9 { "Ada Lovelace".to_string() }
                         else if cuda_compute.minor >= 6 { "Ampere".to_string() }
                         else { "Ampere".to_string() },
                    7 => if cuda_compute.minor >= 5 { "Turing".to_string() }
                         else { "Volta".to_string() },
                    6 => "Pascal".to_string(),
                    5 => "Maxwell".to_string(),
                    _ => format!("SM {}.{}", cuda_compute.major, cuda_compute.minor),
                };
            }

            let total_cuda_cores = device.num_cores().unwrap_or(0);
            let cores_per_sm = match gpu.compute_major {
                9 => 128,
                8 => 128,
                7 => 64,
                6 => 128,
                5 => 128,
                _ => 128,
            };

            gpu.cuda_cores = total_cuda_cores;
            gpu.sm_count = if cores_per_sm > 0 { total_cuda_cores / cores_per_sm } else { 0 };

            let (_, tensor_per_sm, l1_per_sm, l2_kb, bus_width, mem_clock, gpu_clock) =
                get_gpu_specs(&gpu.name, gpu.compute_major, gpu.compute_minor, gpu.sm_count);

            gpu.tensor_cores = gpu.sm_count * tensor_per_sm;
            gpu.l1_cache_per_sm_kb = l1_per_sm;
            gpu.l2_cache_kb = l2_kb;
            gpu.memory_bus_width = bus_width;
            gpu.memory_clock_mhz = mem_clock;
            gpu.gpu_clock_mhz = gpu_clock;

            if let Ok(max_clock) = device.max_clock_info(nvml_wrapper::enum_wrappers::device::Clock::Graphics) {
                gpu.gpu_clock_mhz = max_clock;
            }
            if let Ok(max_mem_clock) = device.max_clock_info(nvml_wrapper::enum_wrappers::device::Clock::Memory) {
                gpu.memory_clock_mhz = max_mem_clock;
            }

            if let Ok(pcie_link) = device.current_pcie_link_gen() {
                gpu.pcie_gen = pcie_link;
            }
            if let Ok(pcie_width) = device.current_pcie_link_width() {
                gpu.pcie_width = pcie_width;
            }

            if gpu.history.len() >= HISTORY_SIZE {
                gpu.history.pop_front();
            }
            gpu.history.push_back(gpu.utilization_percent as f32);
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn collect_gpu(&mut self, app: &mut App) {
        app.gpus.clear();
    }

    fn collect_network(&mut self, app: &mut App, interval_secs: f64) {
        let mut current: HashMap<String, (u64, u64)> = HashMap::new();
        let mut total_rx = 0u64;
        let mut total_tx = 0u64;

        for (name, data) in self.networks.iter() {
            if name == "lo" || name.starts_with("docker") || name.starts_with("veth") {
                continue;
            }
            let rx = data.total_received();
            let tx = data.total_transmitted();
            current.insert(name.clone(), (rx, tx));
            total_rx += rx;
            total_tx += tx;
        }

        if let Some(ref prev) = self.prev_net {
            let prev_rx: u64 = prev.values().map(|(rx, _)| rx).sum();
            let prev_tx: u64 = prev.values().map(|(_, tx)| tx).sum();

            let rx_delta = total_rx.saturating_sub(prev_rx);
            let tx_delta = total_tx.saturating_sub(prev_tx);

            app.network.rx_bytes_per_sec = rx_delta as f64 / interval_secs;
            app.network.tx_bytes_per_sec = tx_delta as f64 / interval_secs;
        }

        self.prev_net = Some(current);
    }

    fn collect_disk(&mut self, app: &mut App, interval_secs: f64) {
        let current = parse_diskstats();

        if let Some(ref prev) = self.prev_disk {
            let mut total_read_delta = 0u64;
            let mut total_write_delta = 0u64;

            for (device, (sectors_read, sectors_written)) in &current {
                if let Some((prev_read, prev_write)) = prev.get(device) {
                    total_read_delta += sectors_read.saturating_sub(*prev_read);
                    total_write_delta += sectors_written.saturating_sub(*prev_write);
                }
            }

            app.disk.read_bytes_per_sec = (total_read_delta as f64 * 512.0) / interval_secs;
            app.disk.write_bytes_per_sec = (total_write_delta as f64 * 512.0) / interval_secs;
        }

        self.prev_disk = Some(current);
    }

    fn collect_battery(&self, app: &mut App) {
        let battery_path = std::path::Path::new("/sys/class/power_supply");
        if !battery_path.exists() {
            app.battery.present = false;
            return;
        }

        if let Ok(entries) = std::fs::read_dir(battery_path) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("BAT") {
                    let bat_path = entry.path();

                    if let Ok(capacity) = std::fs::read_to_string(bat_path.join("capacity")) {
                        app.battery.present = true;
                        app.battery.percent = capacity.trim().parse().unwrap_or(0);
                    }

                    if let Ok(status) = std::fs::read_to_string(bat_path.join("status")) {
                        app.battery.charging = status.trim() == "Charging";
                    }

                    return;
                }
            }
        }

        app.battery.present = false;
    }

    fn collect_processes(&mut self, app: &mut App) {
        self.sys.refresh_processes_specifics(
            sysinfo::ProcessesToUpdate::All,
            true,
            ProcessRefreshKind::everything(),
        );

        let gpu_processes = self.get_gpu_processes();

        app.processes.clear();
        let total_memory = self.sys.total_memory() as f64;

        for (pid, process) in self.sys.processes() {
            let pid_u32 = pid.as_u32();
            let gpu_mem = gpu_processes.get(&pid_u32).copied();

            app.processes.push(ProcessInfo {
                pid: pid_u32,
                name: process.name().to_string_lossy().to_string(),
                command: process
                    .cmd()
                    .iter()
                    .map(|s| s.to_string_lossy().to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
                cpu_percent: process.cpu_usage(),
                memory_percent: (process.memory() as f64 / total_memory * 100.0) as f32,
                memory_bytes: process.memory(),
                gpu_memory_bytes: gpu_mem,
            });
        }

        app.sort_processes();

        if !app.processes.is_empty() && app.selected_process >= app.processes.len() {
            app.selected_process = app.processes.len() - 1;
        }
    }

    #[cfg(target_os = "linux")]
    fn get_gpu_processes(&self) -> HashMap<u32, u64> {
        let mut result = HashMap::new();

        let Some(ref nvml) = self.nvml else {
            return result;
        };

        let device_count = nvml.device_count().unwrap_or(0);
        for i in 0..device_count {
            let Ok(device) = nvml.device_by_index(i) else {
                continue;
            };

            if let Ok(processes) = device.running_compute_processes() {
                for proc in processes {
                    let pid = proc.pid;
                    let mem = match proc.used_gpu_memory {
                        UsedGpuMemory::Used(bytes) => bytes,
                        UsedGpuMemory::Unavailable => 0,
                    };

                    result.entry(pid)
                        .and_modify(|m| *m += mem)
                        .or_insert(mem);
                }
            }

            if let Ok(processes) = device.running_graphics_processes() {
                for proc in processes {
                    let pid = proc.pid;
                    let mem = match proc.used_gpu_memory {
                        UsedGpuMemory::Used(bytes) => bytes,
                        UsedGpuMemory::Unavailable => 0,
                    };

                    result.entry(pid)
                        .and_modify(|m| *m += mem)
                        .or_insert(mem);
                }
            }
        }

        result
    }

    #[cfg(not(target_os = "linux"))]
    fn get_gpu_processes(&self) -> HashMap<u32, u64> {
        HashMap::new()
    }
}

#[cfg(target_os = "linux")]
fn get_gpu_specs(name: &str, compute_major: u32, compute_minor: u32, sm_count: u32) -> (u32, u32, u32, u32, u32, u32, u32) {
    let name_lower = name.to_lowercase();

    let (cores_per_sm, tensor_per_sm, l1_per_sm) = match compute_major {
        9 => (128, 4, 128),
        8 if compute_minor >= 9 => (128, 4, 128),
        8 if compute_minor >= 6 => (128, 4, 128),
        8 => (64, 4, 192),
        7 if compute_minor >= 5 => (64, 8, 96),
        7 => (64, 8, 128),
        6 => (128, 0, 48),
        _ => (128, 0, 48),
    };

    let (l2_kb, bus_width, mem_clock, gpu_clock) = if name_lower.contains("4090") {
        (73728, 384, 10501, 2520)
    } else if name_lower.contains("4080 super") {
        (65536, 256, 11520, 2550)
    } else if name_lower.contains("4080") {
        (65536, 256, 11520, 2505)
    } else if name_lower.contains("4070 ti super") {
        (49152, 256, 10501, 2610)
    } else if name_lower.contains("4070 ti") {
        (49152, 192, 10501, 2610)
    } else if name_lower.contains("4070 super") {
        (49152, 192, 10501, 2475)
    } else if name_lower.contains("4070") {
        (36864, 192, 10501, 2475)
    } else if name_lower.contains("4060 ti") {
        (32768, 128, 9001, 2535)
    } else if name_lower.contains("4060") {
        (24576, 128, 8501, 2460)
    } else if name_lower.contains("3090 ti") {
        (6144, 384, 10752, 1860)
    } else if name_lower.contains("3090") {
        (6144, 384, 9751, 1695)
    } else if name_lower.contains("3080 ti") {
        (6144, 384, 9501, 1665)
    } else if name_lower.contains("3080") {
        (5120, 320, 9501, 1710)
    } else if name_lower.contains("3070 ti") {
        (4096, 256, 9501, 1770)
    } else if name_lower.contains("3070") {
        (4096, 256, 7001, 1725)
    } else if name_lower.contains("3060 ti") {
        (3072, 256, 7001, 1670)
    } else if name_lower.contains("3060") {
        (2560, 192, 7501, 1777)
    } else if name_lower.contains("a100") {
        (40960, 5120, 1215, 1410)
    } else if name_lower.contains("h100") {
        (51200, 5120, 2619, 1980)
    } else if name_lower.contains("a6000") {
        (6144, 384, 8001, 1800)
    } else if name_lower.contains("a5000") {
        (4096, 256, 8001, 1695)
    } else if name_lower.contains("a4000") {
        (3072, 256, 7001, 1560)
    } else {
        let l2 = sm_count * 64;
        let bus = if sm_count > 60 { 384 } else if sm_count > 40 { 256 } else { 192 };
        (l2, bus, 8000, 1800)
    };

    (cores_per_sm, tensor_per_sm, l1_per_sm, l2_kb, bus_width, mem_clock, gpu_clock)
}

fn parse_diskstats() -> HashMap<String, (u64, u64)> {
    let mut result = HashMap::new();

    let Ok(file) = File::open("/proc/diskstats") else {
        return result;
    };

    let reader = BufReader::new(file);
    for line in reader.lines().map_while(Result::ok) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 14 {
            continue;
        }

        let device = parts[2];
        if device.starts_with("loop")
            || device.starts_with("ram")
            || device.starts_with("dm-")
            || device.contains('p') && device.chars().last().map(|c| c.is_ascii_digit()).unwrap_or(false)
        {
            continue;
        }

        let is_partition = device.chars().last().map(|c| c.is_ascii_digit()).unwrap_or(false)
            && (device.starts_with("sd") || device.starts_with("hd"));
        if is_partition {
            continue;
        }

        let sectors_read: u64 = parts[5].parse().unwrap_or(0);
        let sectors_written: u64 = parts[9].parse().unwrap_or(0);

        result.insert(device.to_string(), (sectors_read, sectors_written));
    }

    result
}
