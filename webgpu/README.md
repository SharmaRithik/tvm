# WebGPU Matmul Shader Generation

Generate optimized WGSL compute shaders for FP32 matrix multiplication using TVM.

## Build

```bash
./webgpu/build.sh
```

Requires: emscripten (`emcc`), node, npm, and a native TVM build with LLVM.

## Test

```bash
./webgpu/test.sh
```

## Generate Shaders

```bash
# Single variant (default: optimized)
python webgpu/generate_matmul.py

# All variants to files
python webgpu/generate_matmul.py --all --outdir webgpu/generated-shaders
```

### Variants

| Variant | Threads | Register tile | Shared mem | Description |
|---------|---------|--------------|------------|-------------|
| `naive` | 16x16 | 1x1 | No | Baseline |
| `optimized` | 8x8 | 4x4 | Yes | Default optimized |
| `large_tile` | 16x16 | 4x4 | Yes | Larger output tile |
| `deep_k` | 8x8 | 4x4 | Yes | Deeper reduction |
| `wide_register` | 4x4 | 8x8 | Yes | More work per thread |
| `tall_register` | 16x8 | 8x2 | Yes | Tall micro-kernel |

Pre-generated shaders for 1024x1024x1024 are in `generated-shaders/`.
