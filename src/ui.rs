use crate::app::{App, GpuVendor, KillState, SortColumn};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Sparkline, Table},
    Frame,
};

const DIM: Color = Color::DarkGray;
const BRIGHT: Color = Color::White;

// Braille/block characters for mini sparklines
const SPARK_CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

pub fn draw(frame: &mut Frame, app: &App) {
    let area = frame.area();

    // Main layout: left column (stats) | right column (processes)
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50),  // Left: system stats
            Constraint::Percentage(50),  // Right: processes
        ])
        .split(area);

    let left_width = main_chunks[0].width.saturating_sub(2); // account for borders

    // Calculate CPU layout dynamically
    // Each core entry: "NN ▓▓▓▓▓▓▓▓ XXX%" ~= 18-20 chars, use 20 as safe minimum
    let core_entry_width = 20u16;
    let num_cores = app.cpu.cores.len().max(1);
    let cols = (left_width / core_entry_width).max(1) as usize;
    let cols = cols.min(4); // cap at 4 columns
    let rows = (num_cores + cols - 1) / cols; // ceiling division
    let cpu_height = (rows as u16 + 2).max(4); // +2 for borders, min 4

    // GPU section
    let gpu_count = app.gpus.len();
    let gpu_height = if gpu_count > 0 { 11 * gpu_count as u16 } else { 0 };

    // Calculate remaining space for GPU info
    let fixed_heights = cpu_height + 6 + gpu_height + 3 + 1; // CPU + Mem + GPU + Net/Disk + Status
    let total_height = main_chunks[0].height;
    let extra_space = total_height.saturating_sub(fixed_heights);

    // If we have extra space and GPUs, show GPU metadata
    let gpu_info_height = if gpu_count > 0 && extra_space >= 4 {
        extra_space.min(11) // cap at 11 lines for GPU specs (9 lines + 2 borders)
    } else {
        0
    };

    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(cpu_height),      // CPU
            Constraint::Length(6),               // Memory
            Constraint::Length(gpu_height),      // GPU stats
            Constraint::Length(gpu_info_height), // GPU metadata (if space)
            Constraint::Length(3),               // Network/Disk
            Constraint::Min(0),                  // Absorb remaining
            Constraint::Length(1),               // Status bar
        ])
        .split(main_chunks[0]);

    draw_cpu_combined(frame, app, left_chunks[0]);
    draw_memory_graph(frame, app, left_chunks[1]);
    if gpu_count > 0 {
        draw_gpus_verbose(frame, app, left_chunks[2]);
        if gpu_info_height > 0 {
            draw_gpu_info(frame, app, left_chunks[3]);
        }
    }
    draw_network_disk(frame, app, left_chunks[4]);
    draw_status_bar(frame, app, left_chunks[6]);

    // Right column: processes (full height)
    draw_processes(frame, app, main_chunks[1]);
}

fn draw_cpu_combined(frame: &mut Frame, app: &App, area: Rect) {
    let temp_str = app.cpu.temperature_c
        .map(|t| format!("{}C", t as u32))
        .unwrap_or_else(|| "-".to_string());

    let title = format!(
        " CPU {:.1}% | {} | {} cores ",
        app.cpu.usage_percent,
        temp_str,
        app.cpu.core_count
    );

    let block = Block::default()
        .title(Span::styled(title, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(DIM));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.height == 0 || inner.width == 0 {
        return;
    }

    let num_cores = app.cpu.cores.len();
    if num_cores == 0 {
        return;
    }

    // Dynamic columns based on available width
    // Each core entry: "NN ▓▓▓▓▓▓▓▓ XXX%" = ~18 chars, use 19 with spacing
    let core_entry_width = 19usize;
    let cols = (inner.width as usize / core_entry_width).max(1).min(4);
    let rows = (num_cores + cols - 1) / cols; // ceiling division to fit all cores
    let col_width = inner.width as usize / cols;

    let mut lines: Vec<Line> = Vec::new();

    for row in 0..rows {
        if row >= inner.height as usize {
            break; // Don't exceed available height
        }

        let mut spans: Vec<Span> = Vec::new();

        for col in 0..cols {
            // Vertical traversal: core index = col * rows + row
            let core_idx = col * rows + row;

            if core_idx < num_cores {
                let core = &app.cpu.cores[core_idx];
                let usage = core.usage_percent;
                let color = usage_color(usage);

                // Mini bar using block chars (like btop)
                let bar_width = 8;
                let filled = ((usage / 100.0) * bar_width as f32).round() as usize;
                let bar: String = (0..bar_width).map(|i| {
                    if i < filled { '▓' } else { '░' }
                }).collect();

                // Format: "NN ▓▓▓▓░░░░ XXX%"
                let core_str = format!("{:>2} {} {:>3.0}%", core_idx, bar, usage);

                // Pad to column width
                let padded = format!("{:<width$}", core_str, width = col_width);
                spans.push(Span::styled(padded, Style::default().fg(color)));
            } else {
                spans.push(Span::raw(" ".repeat(col_width)));
            }
        }

        lines.push(Line::from(spans));
    }

    let paragraph = Paragraph::new(lines);
    frame.render_widget(paragraph, inner);
}

fn draw_memory_graph(frame: &mut Frame, app: &App, area: Rect) {
    let mem_percent = if app.memory.total_bytes > 0 {
        app.memory.used_bytes as f64 / app.memory.total_bytes as f64 * 100.0
    } else {
        0.0
    };

    let used_gb = app.memory.used_bytes as f64 / 1_073_741_824.0;
    let total_gb = app.memory.total_bytes as f64 / 1_073_741_824.0;

    let title = format!(" Memory: {:.1}/{:.1} GB ({:.1}%) ", used_gb, total_gb, mem_percent);

    let block = Block::default()
        .title(Span::styled(title, Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(DIM));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let data: Vec<u64> = app.memory.history.iter().map(|v| *v as u64).collect();
    let sparkline = Sparkline::default()
        .data(&data)
        .max(100)
        .style(Style::default().fg(Color::Magenta));
    frame.render_widget(sparkline, inner);
}

fn draw_gpus_verbose(frame: &mut Frame, app: &App, area: Rect) {
    let gpu_count = app.gpus.len();
    if gpu_count == 0 {
        return;
    }

    let height_per_gpu = area.height / gpu_count as u16;
    let constraints: Vec<Constraint> = app.gpus.iter()
        .map(|_| Constraint::Length(height_per_gpu))
        .collect();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    for (i, gpu) in app.gpus.iter().enumerate() {
        draw_gpu_verbose(frame, gpu, chunks[i]);
    }
}

fn draw_gpu_verbose(frame: &mut Frame, gpu: &crate::app::GpuData, area: Rect) {
    // Different title format for Apple vs NVIDIA
    let title = match gpu.vendor {
        GpuVendor::Apple => format!(
            " GPU {} | {} | {} ",
            gpu.index,
            gpu.name,
            gpu.driver_version  // Metal family
        ),
        GpuVendor::Nvidia => format!(
            " GPU {} | {} | Driver {} | CUDA {} ",
            gpu.index,
            gpu.name,
            gpu.driver_version,
            gpu.cuda_version
        ),
    };

    let block = Block::default()
        .title(Span::styled(title, Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(DIM));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.height < 2 {
        return;
    }

    // Split: top row for stats, bottom for graph
    let inner_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),  // Stats row (2 lines)
            Constraint::Min(1),     // Graph (remaining space)
        ])
        .split(inner);

    // Calculate stats
    let mem_used_gb = gpu.memory_used_bytes as f64 / 1_073_741_824.0;
    let mem_total_gb = gpu.memory_total_bytes as f64 / 1_073_741_824.0;
    let mem_percent = if gpu.memory_total_bytes > 0 {
        (gpu.memory_used_bytes as f64 / gpu.memory_total_bytes as f64 * 100.0) as u32
    } else {
        0
    };

    // Memory label differs: VRAM for NVIDIA, Mem for Apple unified
    let mem_label = if gpu.unified_memory { "Mem" } else { "VRAM" };

    // Build stats as horizontal spans on two lines
    let mut line1_spans = vec![
        Span::styled("Util ", Style::default().fg(DIM)),
        Span::styled(format!("{:>3}%", gpu.utilization_percent), Style::default().fg(usage_color(gpu.utilization_percent as f32))),
        Span::raw("  "),
    ];

    // Temperature
    if gpu.temperature_c > 0 || gpu.vendor == GpuVendor::Nvidia {
        line1_spans.push(Span::styled("Temp ", Style::default().fg(DIM)));
        line1_spans.push(Span::styled(format!("{:>3}C", gpu.temperature_c), Style::default().fg(temp_color(gpu.temperature_c))));
        line1_spans.push(Span::raw("  "));
    }

    // Memory
    line1_spans.push(Span::styled(format!("{} ", mem_label), Style::default().fg(DIM)));
    line1_spans.push(Span::styled(
        format!("{:.1}/{:.1}GB", mem_used_gb, mem_total_gb),
        Style::default().fg(usage_color(mem_percent as f32))
    ));

    let mut line2_spans = Vec::new();

    // Power
    if gpu.power_usage_w > 0 {
        let power_percent = if gpu.power_limit_w > 0 {
            (gpu.power_usage_w as f64 / gpu.power_limit_w as f64 * 100.0) as u32
        } else {
            0
        };
        line2_spans.push(Span::styled("Power ", Style::default().fg(DIM)));
        line2_spans.push(Span::styled(
            format!("{:>3}/{:>3}W", gpu.power_usage_w, gpu.power_limit_w),
            Style::default().fg(usage_color(power_percent as f32))
        ));
        line2_spans.push(Span::raw("  "));
    } else if gpu.power_limit_w > 0 {
        // Show TDP estimate for Apple
        line2_spans.push(Span::styled("TDP ", Style::default().fg(DIM)));
        line2_spans.push(Span::styled(format!("~{}W", gpu.power_limit_w), Style::default().fg(BRIGHT)));
        line2_spans.push(Span::raw("  "));
    }

    // Fan (only for NVIDIA)
    if gpu.vendor == GpuVendor::Nvidia && gpu.fan_speed_percent > 0 {
        line2_spans.push(Span::styled("Fan ", Style::default().fg(DIM)));
        line2_spans.push(Span::styled(format!("{:>3}%", gpu.fan_speed_percent), Style::default().fg(BRIGHT)));
    }

    let lines = vec![
        Line::from(line1_spans),
        Line::from(line2_spans),
    ];

    let stats = Paragraph::new(lines);
    frame.render_widget(stats, inner_chunks[0]);

    // Graph on bottom
    let data: Vec<u64> = gpu.history.iter().map(|v| *v as u64).collect();
    let sparkline = Sparkline::default()
        .data(&data)
        .max(100)
        .style(Style::default().fg(Color::Green));
    frame.render_widget(sparkline, inner_chunks[1]);
}

fn temp_color(temp: u32) -> Color {
    if temp >= 80 {
        Color::Red
    } else if temp >= 70 {
        Color::LightRed
    } else if temp >= 60 {
        Color::Yellow
    } else {
        Color::Green
    }
}

fn draw_gpu_info(frame: &mut Frame, app: &App, area: Rect) {
    if app.gpus.is_empty() {
        return;
    }

    let block = Block::default()
        .title(Span::styled(" GPU Specs ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(DIM));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.height == 0 {
        return;
    }

    let gpu = &app.gpus[0];

    let lines = match gpu.vendor {
        GpuVendor::Apple => draw_apple_gpu_specs(gpu),
        GpuVendor::Nvidia => draw_nvidia_gpu_specs(gpu),
    };

    // Trim to available height
    let lines: Vec<Line> = lines.into_iter().take(inner.height as usize).collect();

    let paragraph = Paragraph::new(lines);
    frame.render_widget(paragraph, inner);
}

fn draw_nvidia_gpu_specs(gpu: &crate::app::GpuData) -> Vec<Line<'static>> {
    // Format L2 cache nicely
    let l2_str = if gpu.l2_cache_kb >= 1024 {
        format!("{} MB", gpu.l2_cache_kb / 1024)
    } else {
        format!("{} KB", gpu.l2_cache_kb)
    };

    vec![
        Line::from(vec![
            Span::styled("Arch       ", Style::default().fg(DIM)),
            Span::styled(format!("{} (SM {}.{})", gpu.architecture, gpu.compute_major, gpu.compute_minor), Style::default().fg(BRIGHT)),
        ]),
        Line::from(vec![
            Span::styled("SMs        ", Style::default().fg(DIM)),
            Span::styled(format!("{}", gpu.sm_count), Style::default().fg(BRIGHT)),
        ]),
        Line::from(vec![
            Span::styled("CUDA Cores ", Style::default().fg(DIM)),
            Span::styled(format!("{}", gpu.cuda_cores), Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("Tensor     ", Style::default().fg(DIM)),
            Span::styled(format!("{}", gpu.tensor_cores), Style::default().fg(Color::Magenta)),
        ]),
        Line::from(vec![
            Span::styled("L1/SM      ", Style::default().fg(DIM)),
            Span::styled(format!("{} KB", gpu.l1_cache_per_sm_kb), Style::default().fg(BRIGHT)),
        ]),
        Line::from(vec![
            Span::styled("L2 Cache   ", Style::default().fg(DIM)),
            Span::styled(l2_str, Style::default().fg(BRIGHT)),
        ]),
        Line::from(vec![
            Span::styled("Mem Bus    ", Style::default().fg(DIM)),
            Span::styled(format!("{}-bit @ {} MHz", gpu.memory_bus_width, gpu.memory_clock_mhz), Style::default().fg(BRIGHT)),
        ]),
        Line::from(vec![
            Span::styled("GPU Clock  ", Style::default().fg(DIM)),
            Span::styled(format!("{} MHz (boost)", gpu.gpu_clock_mhz), Style::default().fg(BRIGHT)),
        ]),
        Line::from(vec![
            Span::styled("PCIe       ", Style::default().fg(DIM)),
            Span::styled(format!("Gen{} x{}", gpu.pcie_gen, gpu.pcie_width), Style::default().fg(BRIGHT)),
        ]),
    ]
}

fn draw_apple_gpu_specs(gpu: &crate::app::GpuData) -> Vec<Line<'static>> {
    // Apple Silicon GPU uses different terminology:
    // - GPU Cores (not SMs)
    // - Execution Units (EUs) - 16 per GPU core
    // - ALUs - 128 per GPU core (8 per EU)
    // - Neural Engine (separate from GPU)
    // - System Level Cache (SLC) instead of L2
    // - Unified Memory with bandwidth in GB/s

    vec![
        Line::from(vec![
            Span::styled("Arch       ", Style::default().fg(DIM)),
            Span::styled(gpu.architecture.clone(), Style::default().fg(BRIGHT)),
        ]),
        Line::from(vec![
            Span::styled("GPU Cores  ", Style::default().fg(DIM)),
            Span::styled(format!("{}", gpu.gpu_cores), Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("Exec Units ", Style::default().fg(DIM)),
            Span::styled(format!("{} EUs (16/core)", gpu.execution_units), Style::default().fg(BRIGHT)),
        ]),
        Line::from(vec![
            Span::styled("ALUs       ", Style::default().fg(DIM)),
            Span::styled(format!("{} (128/core)", gpu.alu_count), Style::default().fg(BRIGHT)),
        ]),
        Line::from(vec![
            Span::styled("Neural Eng ", Style::default().fg(DIM)),
            Span::styled(format!("{}-core ({} TOPS)", gpu.neural_engine_cores, gpu.neural_engine_tops), Style::default().fg(Color::Magenta)),
        ]),
        Line::from(vec![
            Span::styled("SLC        ", Style::default().fg(DIM)),
            Span::styled(format!("{} MB", gpu.slc_mb), Style::default().fg(BRIGHT)),
        ]),
        Line::from(vec![
            Span::styled("Mem BW     ", Style::default().fg(DIM)),
            Span::styled(format!("{} GB/s", gpu.memory_bandwidth_gbps), Style::default().fg(BRIGHT)),
        ]),
        Line::from(vec![
            Span::styled("Memory     ", Style::default().fg(DIM)),
            Span::styled("Unified LPDDR5x".to_string(), Style::default().fg(BRIGHT)),
        ]),
    ]
}

fn draw_network_disk(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .title(Span::styled(" Network / Disk ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(DIM));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let info = Line::from(vec![
        Span::styled("NET ", Style::default().fg(DIM)),
        Span::styled("↓", Style::default().fg(Color::Green)),
        Span::styled(format_bytes_rate(app.network.rx_bytes_per_sec), Style::default().fg(BRIGHT)),
        Span::raw(" "),
        Span::styled("↑", Style::default().fg(Color::Red)),
        Span::styled(format_bytes_rate(app.network.tx_bytes_per_sec), Style::default().fg(BRIGHT)),
        Span::raw("   "),
        Span::styled("DISK ", Style::default().fg(DIM)),
        Span::styled("R", Style::default().fg(Color::Green)),
        Span::styled(format_bytes_rate(app.disk.read_bytes_per_sec), Style::default().fg(BRIGHT)),
        Span::raw(" "),
        Span::styled("W", Style::default().fg(Color::Red)),
        Span::styled(format_bytes_rate(app.disk.write_bytes_per_sec), Style::default().fg(BRIGHT)),
    ]);
    frame.render_widget(Paragraph::new(info), inner);
}

fn draw_processes(frame: &mut Frame, app: &App, area: Rect) {
    let sort_indicator = |col: SortColumn| {
        if app.sort_column == col { " ▼" } else { "" }
    };

    let title = match app.kill_state {
        KillState::None => format!(
            " Processes [{}] - Sort: {} ",
            app.processes.len(),
            app.sort_column.name()
        ),
        KillState::Confirming(pid) => format!(
            " KILL PID {}? [Enter] confirm / [Esc] cancel ",
            pid
        ),
    };

    let title_color = match app.kill_state {
        KillState::None => Color::Blue,
        KillState::Confirming(_) => Color::Red,
    };

    let block = Block::default()
        .title(Span::styled(title, Style::default().fg(title_color).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(DIM));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.height < 2 {
        return;
    }

    // Header row
    let header = Row::new(vec![
        format!("{:>7}", "PID"),
        format!("{:<20}", "NAME"),
        format!("{:>6}{}", "CPU%", sort_indicator(SortColumn::Cpu)),
        format!("{:>6}{}", "RAM%", sort_indicator(SortColumn::Ram)),
        format!("{:>9}", "RAM"),
        format!("{:>9}{}", "GPU", sort_indicator(SortColumn::Gpu)),
    ])
    .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));

    let rows: Vec<Row> = app
        .processes
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let is_selected = i == app.selected_process;
            let style = if is_selected {
                Style::default().bg(Color::DarkGray).fg(Color::White)
            } else {
                Style::default().fg(BRIGHT)
            };

            let vram_str = p.gpu_memory_bytes
                .map(|b| format_bytes(b))
                .unwrap_or_else(|| "-".to_string());

            Row::new(vec![
                format!("{:>7}", p.pid),
                format!("{:<20}", truncate(&p.name, 20)),
                format!("{:>6.1}", p.cpu_percent),
                format!("{:>6.1}", p.memory_percent),
                format!("{:>9}", format_bytes(p.memory_bytes)),
                format!("{:>9}", vram_str),
            ])
            .style(style)
        })
        .collect();

    let widths = [
        Constraint::Length(8),
        Constraint::Length(21),
        Constraint::Length(9),
        Constraint::Length(9),
        Constraint::Length(10),
        Constraint::Length(12),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .row_highlight_style(Style::default());

    frame.render_widget(table, inner);
}

fn draw_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    let mut spans = vec![
        Span::styled(" Q", Style::default().fg(Color::Cyan)),
        Span::raw(" quit  "),
        Span::styled("↑↓", Style::default().fg(Color::Cyan)),
        Span::raw(" select  "),
        Span::styled("←→", Style::default().fg(Color::Cyan)),
        Span::raw(" sort  "),
        Span::styled("K", Style::default().fg(Color::Cyan)),
        Span::raw(" kill"),
    ];

    // Add battery if present
    if app.battery.present {
        let battery_icon = if app.battery.charging { "+" } else { "" };
        spans.push(Span::raw("  "));
        spans.push(Span::styled(
            format!("BAT {}%{}", app.battery.percent, battery_icon),
            Style::default().fg(battery_color(app.battery.percent)),
        ));
    }

    let status = Paragraph::new(Line::from(spans));
    frame.render_widget(status, area);
}

// Color gradient from green (0%) -> yellow (50%) -> red (100%)
fn usage_color(percent: f32) -> Color {
    if percent >= 90.0 {
        Color::Red
    } else if percent >= 70.0 {
        Color::LightRed
    } else if percent >= 50.0 {
        Color::Yellow
    } else if percent >= 30.0 {
        Color::LightYellow
    } else {
        Color::Green
    }
}

fn battery_color(percent: u8) -> Color {
    if percent <= 20 {
        Color::Red
    } else if percent <= 50 {
        Color::Yellow
    } else {
        Color::Green
    }
}

// Fixed-width: "999.9 XB/s" = 10 chars
fn format_bytes_rate(bytes_per_sec: f64) -> String {
    if bytes_per_sec >= 1_000_000_000.0 {
        format!("{:>5.1} GB/s", bytes_per_sec / 1_000_000_000.0)
    } else if bytes_per_sec >= 1_000_000.0 {
        format!("{:>5.1} MB/s", bytes_per_sec / 1_000_000.0)
    } else if bytes_per_sec >= 1_000.0 {
        format!("{:>5.1} KB/s", bytes_per_sec / 1_000.0)
    } else {
        format!("{:>5.0}  B/s", bytes_per_sec)
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}
