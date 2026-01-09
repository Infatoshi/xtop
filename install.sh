#!/bin/bash
set -e

echo "Installing xtop..."

# Check for Rust
if ! command -v cargo &> /dev/null; then
    echo "Rust not found. Installing via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Clone and build
TMPDIR=$(mktemp -d)
cd "$TMPDIR"
git clone --depth 1 https://github.com/infatoshi/xtop.git
cd xtop
cargo install --path .

# Cleanup
rm -rf "$TMPDIR"

echo "xtop installed successfully!"
echo "Run 'xtop' to start."
