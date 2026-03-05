#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TVM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEB_DIR="$TVM_ROOT/web"

if ! command -v emcc &>/dev/null; then
    if [ -d "$HOME/emsdk" ]; then
        source "$HOME/emsdk/emsdk_env.sh" 2>/dev/null
    else
        echo "Error: emcc not found. Install emsdk first:"
        echo "  git clone https://github.com/emscripten-core/emsdk.git ~/emsdk"
        echo "  cd ~/emsdk && ./emsdk install latest && ./emsdk activate latest"
        exit 1
    fi
fi

echo "=== Building WASM runtime ==="
cd "$WEB_DIR"
make clean
make

for f in dist/wasm/tvmjs_runtime.wasi.js src/tvmjs_runtime_wasi.js; do
    if [ -f "$f" ]; then
        sed -i 's|require("node:fs")|require("fs")|g' "$f"
        sed -i 's|require("node:crypto")|require("crypto")|g' "$f"
        sed -i 's|require("node:path")|require("path")|g' "$f"
    fi
done

echo "=== Installing npm dependencies ==="
npm install --silent

echo "=== Building JS frontend ==="
npm run bundle

echo "=== Build complete ==="
