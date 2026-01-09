use std::collections::VecDeque;

// History buffer size (60 seconds at 500ms = 120 samples)
pub const HISTORY_SIZE: usize = 120;
// Mini history for per-core sparklines (4 bars)
pub const CORE_HISTORY_SIZE: usize = 4;

#[derive(Debug, Clone)]
pub struct CoreData {
    pub usage_percent: f32,
    pub history: VecDeque<f32>,
}

impl Default for CoreData {
    fn default() -> Self {
        Self {
            usage_percent: 0.0,
            history: VecDeque::with_capacity(CORE_HISTORY_SIZE),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CpuData {
    pub model: String,
    pub core_count: usize,
    pub cores: Vec<CoreData>,
    pub usage_percent: f32,
    pub temperature_c: Option<f32>,
    pub frequency_mhz: u64,
    pub history: VecDeque<f32>,
}

impl Default for CpuData {
    fn default() -> Self {
        Self {
            model: String::new(),
            core_count: 0,
            cores: Vec::new(),
            usage_percent: 0.0,
            temperature_c: None,
            frequency_mhz: 0,
            history: VecDeque::with_capacity(HISTORY_SIZE),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryData {
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub swap_total_bytes: u64,
    pub swap_used_bytes: u64,
    pub history: VecDeque<f32>,
}

impl Default for MemoryData {
    fn default() -> Self {
        Self {
            total_bytes: 0,
            used_bytes: 0,
            swap_total_bytes: 0,
            swap_used_bytes: 0,
            history: VecDeque::with_capacity(HISTORY_SIZE),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuData {
    pub index: u32,
    pub name: String,
    pub utilization_percent: u32,
    pub temperature_c: u32,
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
    pub power_usage_w: u32,
    pub power_limit_w: u32,
    pub fan_speed_percent: u32,
    pub driver_version: String,
    pub cuda_version: String,
    pub history: VecDeque<f32>,
    // GPU metadata (cudaDeviceQuery-style info)
    pub architecture: String,
    pub compute_major: u32,
    pub compute_minor: u32,
    pub sm_count: u32,
    pub cuda_cores: u32,
    pub tensor_cores: u32,
    pub l1_cache_per_sm_kb: u32,
    pub l2_cache_kb: u32,
    pub memory_bus_width: u32,
    pub memory_clock_mhz: u32,
    pub gpu_clock_mhz: u32,
    pub pcie_gen: u32,
    pub pcie_width: u32,
}

impl Default for GpuData {
    fn default() -> Self {
        Self {
            index: 0,
            name: String::new(),
            utilization_percent: 0,
            temperature_c: 0,
            memory_used_bytes: 0,
            memory_total_bytes: 0,
            power_usage_w: 0,
            power_limit_w: 0,
            fan_speed_percent: 0,
            driver_version: String::new(),
            cuda_version: String::new(),
            history: VecDeque::with_capacity(HISTORY_SIZE),
            architecture: String::new(),
            compute_major: 0,
            compute_minor: 0,
            sm_count: 0,
            cuda_cores: 0,
            tensor_cores: 0,
            l1_cache_per_sm_kb: 0,
            l2_cache_kb: 0,
            memory_bus_width: 0,
            memory_clock_mhz: 0,
            gpu_clock_mhz: 0,
            pcie_gen: 0,
            pcie_width: 0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct NetworkData {
    pub rx_bytes_per_sec: f64,
    pub tx_bytes_per_sec: f64,
}

#[derive(Debug, Clone, Default)]
pub struct DiskData {
    pub read_bytes_per_sec: f64,
    pub write_bytes_per_sec: f64,
}

#[derive(Debug, Clone, Default)]
pub struct BatteryData {
    pub present: bool,
    pub percent: u8,
    pub charging: bool,
}

#[derive(Debug, Clone)]
pub struct ProcessInfo {
    pub pid: u32,
    pub name: String,
    pub command: String,
    pub cpu_percent: f32,
    pub memory_percent: f32,
    pub memory_bytes: u64,
    pub gpu_memory_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortColumn {
    Cpu,
    Ram,
    Vram,
}

impl SortColumn {
    pub fn next(self) -> Self {
        match self {
            Self::Cpu => Self::Ram,
            Self::Ram => Self::Vram,
            Self::Vram => Self::Cpu,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            Self::Cpu => Self::Vram,
            Self::Ram => Self::Cpu,
            Self::Vram => Self::Ram,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Ram => "RAM",
            Self::Vram => "VRAM",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KillState {
    None,
    Confirming(u32),  // Stores the PID to kill
}

pub struct App {
    pub cpu: CpuData,
    pub memory: MemoryData,
    pub gpus: Vec<GpuData>,
    pub network: NetworkData,
    pub disk: DiskData,
    pub battery: BatteryData,
    pub processes: Vec<ProcessInfo>,
    pub selected_process: usize,
    pub sort_column: SortColumn,
    pub kill_state: KillState,
    pub should_quit: bool,
}

impl Default for App {
    fn default() -> Self {
        Self {
            cpu: CpuData::default(),
            memory: MemoryData::default(),
            gpus: Vec::new(),
            network: NetworkData::default(),
            disk: DiskData::default(),
            battery: BatteryData::default(),
            processes: Vec::new(),
            selected_process: 0,
            sort_column: SortColumn::Cpu,
            kill_state: KillState::None,
            should_quit: false,
        }
    }
}

impl App {
    pub fn sort_processes(&mut self) {
        match self.sort_column {
            SortColumn::Cpu => self.processes.sort_by(|a, b| {
                b.cpu_percent.partial_cmp(&a.cpu_percent).unwrap_or(std::cmp::Ordering::Equal)
            }),
            SortColumn::Ram => self.processes.sort_by(|a, b| {
                b.memory_percent.partial_cmp(&a.memory_percent).unwrap_or(std::cmp::Ordering::Equal)
            }),
            SortColumn::Vram => self.processes.sort_by(|a, b| {
                let a_vram = a.gpu_memory_bytes.unwrap_or(0);
                let b_vram = b.gpu_memory_bytes.unwrap_or(0);
                b_vram.cmp(&a_vram)
            }),
        }
    }

    pub fn selected_pid(&self) -> Option<u32> {
        self.processes.get(self.selected_process).map(|p| p.pid)
    }
}
