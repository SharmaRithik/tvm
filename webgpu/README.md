# WebGPU Matmul WGSL Generation

## Setup

```bash
./webgpu/build.sh
```

## Generate

```bash
python webgpu/generate_matmul.py                          # dlight 1024,2048,4096
python webgpu/generate_matmul.py --all                    # all variants, all sizes
python webgpu/generate_matmul.py --variant naive --sizes 512 1024
python webgpu/generate_matmul.py --stdout --sizes 1024    # print to terminal
```

## Test

```bash
./webgpu/test.sh
```
