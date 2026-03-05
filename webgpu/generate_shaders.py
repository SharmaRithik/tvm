#!/usr/bin/env python3
import argparse
import os
import tvm
from tvm import te
from tvm.s_tir import dlight as dl
from tvm.tir.build import split_host_device_mods, codegen_build

DEVICE_TARGET = tvm.target.Target("webgpu")
HOST_TARGET = tvm.target.Target({"kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"})
FULL_TARGET = DEVICE_TARGET.with_host(HOST_TARGET)

DLIGHT_RULES = [
    dl.gpu.Matmul(),
    dl.gpu.GEMV(),
    dl.gpu.Reduction(),
    dl.gpu.GeneralReduction(),
    dl.gpu.Fallback(),
]


def lower_to_wgsl(mod):
    with tvm.transform.PassContext(opt_level=3):
        mod = tvm.tir.transform.BindTarget(FULL_TARGET)(mod)
        mod = tvm.tir.get_tir_pipeline("default")(mod)
        _, device_mod_dict = split_host_device_mods(mod)
        for target, device_mod in device_mod_dict.items():
            device_mod = tvm.tir.pipeline.finalize_device_passes()(device_mod)
            return codegen_build(device_mod, target).inspect_source("")
    return None


def gen_matmul(M, N, K, dtype):
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    mod = tvm.IRModule.from_expr(te.create_prim_func([A, B, C]))
    with FULL_TARGET:
        mod = dl.ApplyDefaultSchedule(*DLIGHT_RULES)(mod)
    return lower_to_wgsl(mod)


def gen_mvmul(M, K, dtype):
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K,), name="B", dtype=dtype)
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M,), lambda i: te.sum(A[i, k] * B[k], axis=k), name="C")
    mod = tvm.IRModule.from_expr(te.create_prim_func([A, B, C]))
    with FULL_TARGET:
        mod = dl.ApplyDefaultSchedule(*DLIGHT_RULES)(mod)
    return lower_to_wgsl(mod)


DTYPES = {"F32": "float32", "F16": "float16"}

OPS = {
    "matmul": lambda size, dtype: gen_matmul(size, size, size, dtype),
    "mvmul": lambda size, dtype: gen_mvmul(size, size, dtype),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=list(OPS.keys()), default=None)
    parser.add_argument("--dtype", choices=list(DTYPES.keys()), default=None)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--outdir", type=str, default="webgpu/generated-shaders")
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()

    ops = [args.op] if args.op else list(OPS.keys())
    dtypes = [args.dtype] if args.dtype else list(DTYPES.keys())
    os.makedirs(args.outdir, exist_ok=True)

    for op in ops:
        for dt in dtypes:
            print(f"{op}{dt} size={args.size} ...", end=" ", flush=True)
            wgsl = OPS[op](args.size, DTYPES[dt])
            if wgsl is None:
                print("FAILED")
                continue
            if args.stdout:
                print(f"\n{wgsl}")
            else:
                path = os.path.join(args.outdir, f"{op}{dt}.wgsl")
                with open(path, "w") as f:
                    f.write(wgsl)
                print(f"-> {path} ({len(wgsl)} chars)")


if __name__ == "__main__":
    main()
