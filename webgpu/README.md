# WebGPU Matmul Shader Generation

Generate WGSL compute shaders for FP32 matrix multiplication using TVM's compiler.

## Build

```bash
./webgpu/build.sh
```

## Test

```bash
./webgpu/test.sh
```

## Generate Shaders

```bash
# Compiler-optimized (DLight auto-schedule)
python webgpu/generate_matmul.py

# All variants
python webgpu/generate_matmul.py --all --outdir webgpu/generated-shaders
```

### Variants

| Variant | Description |
|---------|-------------|
| `dlight` | TVM DLight compiler-scheduled (shared mem, register tiling, cooperative fetch) |
| `dlight_transB` | Same as above, transposed B layout |
| `naive` | Baseline: 1 element/thread, no shared memory |

Pre-generated shaders for 1024x1024x1024 are in `generated-shaders/`.
