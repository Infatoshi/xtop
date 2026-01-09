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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuVendor {
    #[default]
    Nvidia,
    Apple,
}

#[derive(Debug, Clone)]
pub struct GpuData {
    pub index: u32,
    pub name: String,
    pub vendor: GpuVendor,
    pub utilization_percent: u32,
    pub temperature_c: u32,
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
    pub power_usage_mw: u32,    // milliwatts for precision
    pub power_limit_mw: u32,    // milliwatts
    pub fan_speed_percent: u32,
    pub driver_version: String,  // NVIDIA: driver version, Apple: Metal family
    pub cuda_version: String,    // NVIDIA only
    pub history: VecDeque<f32>,
    pub architecture: String,

    // NVIDIA-specific fields
    pub compute_major: u32,      // CUDA compute capability major
    pub compute_minor: u32,      // CUDA compute capability minor
    pub sm_count: u32,           // Streaming Multiprocessors
    pub cuda_cores: u32,         // CUDA cores total
    pub tensor_cores: u32,       // Tensor cores total
    pub l1_cache_per_sm_kb: u32, // L1 cache per SM
    pub l2_cache_kb: u32,        // L2 cache total
    pub memory_bus_width: u32,   // Memory bus width in bits
    pub memory_clock_mhz: u32,      // max memory clock
    pub gpu_clock_mhz: u32,         // max GPU clock
    pub current_gpu_clock_mhz: u32, // current GPU clock (varies)
    pub current_mem_clock_mhz: u32, // current memory clock (varies)
    pub pcie_gen: u32,
    pub pcie_width: u32,

    // Apple-specific fields
    pub gpu_cores: u32,              // Apple GPU cores
    pub execution_units: u32,        // EUs (16 per GPU core)
    pub alu_count: u32,              // ALUs (128 per GPU core)
    pub neural_engine_cores: u32,    // Neural Engine cores (separate from GPU)
    pub neural_engine_tops: u32,     // Neural Engine performance in TOPS
    pub slc_mb: u32,                 // System Level Cache in MB
    pub memory_bandwidth_gbps: u32,  // Memory bandwidth in GB/s
    pub unified_memory: bool,        // True for Apple Silicon
}

impl Default for GpuData {
    fn default() -> Self {
        Self {
            index: 0,
            name: String::new(),
            vendor: GpuVendor::default(),
            utilization_percent: 0,
            temperature_c: 0,
            memory_used_bytes: 0,
            memory_total_bytes: 0,
            power_usage_mw: 0,
            power_limit_mw: 0,
            fan_speed_percent: 0,
            driver_version: String::new(),
            cuda_version: String::new(),
            history: VecDeque::with_capacity(HISTORY_SIZE),
            architecture: String::new(),
            // NVIDIA fields
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
            current_gpu_clock_mhz: 0,
            current_mem_clock_mhz: 0,
            pcie_gen: 0,
            pcie_width: 0,
            // Apple fields
            gpu_cores: 0,
            execution_units: 0,
            alu_count: 0,
            neural_engine_cores: 0,
            neural_engine_tops: 0,
            slc_mb: 0,
            memory_bandwidth_gbps: 0,
            unified_memory: false,
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
    Gpu,
    Vram,
}

impl SortColumn {
    pub fn next(self) -> Self {
        match self {
            Self::Cpu => Self::Ram,
            Self::Ram => Self::Gpu,
            Self::Gpu => Self::Vram,
            Self::Vram => Self::Cpu,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            Self::Cpu => Self::Vram,
            Self::Ram => Self::Cpu,
            Self::Gpu => Self::Ram,
            Self::Vram => Self::Gpu,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Ram => "RAM",
            Self::Gpu => "GPU",
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
            SortColumn::Gpu | SortColumn::Vram => self.processes.sort_by(|a, b| {
                let a_gpu = a.gpu_memory_bytes.unwrap_or(0);
                let b_gpu = b.gpu_memory_bytes.unwrap_or(0);
                b_gpu.cmp(&a_gpu)
            }),
        }
    }

    pub fn selected_pid(&self) -> Option<u32> {
        self.processes.get(self.selected_process).map(|p| p.pid)
    }
}
