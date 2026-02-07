#!/usr/bin/env python3
"""Generate WebGPU WGSL shader for FP32 matmul using TVM.

Uses TVM's DLight compiler rules to automatically schedule the matmul
for WebGPU — no hand-tuned tiling parameters needed.

Usage:
    python generate_matmul.py                      # compiler-scheduled matmul
    python generate_matmul.py --all --outdir shaders
"""

import argparse
import os
import tvm
from tvm import te, tir
from tvm.dlight.gpu import Matmul, Fallback
from tvm.dlight.base import ApplyDefaultSchedule


# ---------------------------------------------------------------------------
# Matmul computation
# ---------------------------------------------------------------------------

def _matmul_compute(M, N, K, transB=False):
    """Define matmul: C[M,N] = A[M,K] @ B[K,N] (or B[N,K]^T)."""
    A = te.placeholder((M, K), name="A", dtype="float32")
    k = te.reduce_axis((0, K), name="k")
    if transB:
        B = te.placeholder((N, K), name="B", dtype="float32")
        C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[j, k], axis=k), name="C")
    else:
        B = te.placeholder((K, N), name="B", dtype="float32")
        C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    return A, B, C


def _extract_wgsl(built_mod):
    """Extract WGSL source from a built TVM module."""
    for m in built_mod.imports_:
        if m.kind == "webgpu":
            return m.inspect_source()
    return built_mod.inspect_source()


# ---------------------------------------------------------------------------
# DLight compiler-driven scheduling
# ---------------------------------------------------------------------------

def _apply_dlight(mod, target):
    """Let TVM's DLight rules automatically schedule for WebGPU."""
    with target:
        return ApplyDefaultSchedule(Matmul(), Fallback())(mod)


def schedule_dlight(M, N, K, target):
    """Compiler-scheduled matmul: DLight Matmul rule picks tiling automatically."""
    A, B, C = _matmul_compute(M, N, K)
    mod = tvm.IRModule.from_expr(
        te.create_prim_func([A, B, C]).with_attr("target", target)
    )
    return _apply_dlight(mod, target)


def schedule_dlight_transB(M, N, K, target):
    """Compiler-scheduled matmul with transposed B layout (B is NxK)."""
    A, B, C = _matmul_compute(M, N, K, transB=True)
    mod = tvm.IRModule.from_expr(
        te.create_prim_func([A, B, C]).with_attr("target", target)
    )
    return _apply_dlight(mod, target)


def schedule_naive(M, N, K, target):
    """Minimal schedule: one element per thread, no shared memory.
    Serves as a baseline to compare the compiler-optimized version against.
    """
    A, B, C = _matmul_compute(M, N, K)
    mod = tvm.IRModule.from_expr(te.create_prim_func([A, B, C]))
    sch = tir.Schedule(mod)
    block = sch.get_block("C")
    i, j, k = sch.get_loops(block)
    tile = 16
    i0, i1 = sch.split(i, [None, tile])
    j0, j1 = sch.split(j, [None, tile])
    sch.reorder(i0, j0, i1, j1)
    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")
    sch.bind(i1, "threadIdx.y")
    sch.bind(j1, "threadIdx.x")
    return sch.mod


VARIANTS = {
    "dlight":       ("Compiler-scheduled (DLight Matmul rule)", schedule_dlight),
    "dlight_transB":("Compiler-scheduled, transposed B layout", schedule_dlight_transB),
    "naive":        ("Baseline: 1 elem/thread, no shared mem", schedule_naive),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(variant, M=1024, N=1024, K=1024):
    """Generate WGSL for a given variant."""
    desc, sched_fn = VARIANTS[variant]
    target = tvm.target.Target("webgpu")
    mod = sched_fn(M, N, K, target)
    built = tvm.build(mod, target=target)
    return _extract_wgsl(built), desc


def main():
    parser = argparse.ArgumentParser(description="Generate matmul FP32 WGSL shaders via TVM")
    parser.add_argument("--variant", choices=list(VARIANTS.keys()), default="dlight")
    parser.add_argument("--all", action="store_true", help="Generate all variants")
    parser.add_argument("--outdir", type=str, default=None, help="Write shaders to directory")
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--K", type=int, default=1024)
    args = parser.parse_args()

    variants = list(VARIANTS.keys()) if args.all else [args.variant]

    for name in variants:
        wgsl, desc = generate(name, args.M, args.N, args.K)
        header = (
            f"// Variant: {name}\n"
            f"// {desc}\n"
            f"// Matrix: C[{args.M},{args.N}] = A[{args.M},{args.K}] @ B[{args.K},{args.N}]\n"
        )

        if args.outdir:
            os.makedirs(args.outdir, exist_ok=True)
            path = os.path.join(args.outdir, f"matmul_{name}.wgsl")
            with open(path, "w") as f:
                f.write(header + wgsl)
            print(f"  {name:20s} -> {path}")
        else:
            print(f"{'='*70}")
            print(f"Variant: {name} — {desc}")
            print(f"{'='*70}")
            print(wgsl)
            print()


if __name__ == "__main__":
    main()
