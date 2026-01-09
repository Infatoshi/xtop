# xtop

Cross-platform terminal system monitor with GPU support.

## Install

### From source (recommended)

Requires [Rust](https://rustup.rs/).

```bash
git clone https://github.com/infatoshi/xtop.git
cd xtop
cargo install --path .
```

### One-liner

```bash
curl -sSL https://raw.githubusercontent.com/infatoshi/xtop/main/install.sh | bash
```

## Usage

```bash
xtop
```

### Controls

- `j/k` or arrow keys: navigate processes
- `h/l` or left/right: change sort column (CPU/RAM/VRAM)
- `K`: kill selected process (with confirmation)
- `q`: quit

## Features

- CPU usage per core with history graphs
- Memory and swap usage
- NVIDIA GPU monitoring (Linux only, requires NVIDIA drivers)
- Network and disk I/O
- Process list with CPU/RAM/VRAM sorting
- Process kill functionality

## Requirements

- Linux or macOS
- NVIDIA drivers (optional, for GPU monitoring on Linux)
