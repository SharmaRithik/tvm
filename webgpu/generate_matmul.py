#!/usr/bin/env python3
"""Generate WebGPU WGSL shader for FP32 matmul using TVM.

Usage:
    python generate_matmul.py                      # default 1024x1024x1024
    python generate_matmul.py --variant optimized   # specific variant
    python generate_matmul.py --all --outdir shaders # all variants to files
"""

import argparse
import os
import tvm
from tvm import te, tir


# ---------------------------------------------------------------------------
# Matmul computation (shared by all variants)
# ---------------------------------------------------------------------------

def _matmul_compute(M, N, K):
    """Define C[M,N] = A[M,K] @ B[K,N]."""
    A = te.placeholder((M, K), name="A", dtype="float32")
    B = te.placeholder((K, N), name="B", dtype="float32")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    return A, B, C


def _extract_wgsl(built_mod):
    """Extract WGSL source from a built TVM module."""
    for m in built_mod.imports_:
        if m.kind == "webgpu":
            return m.inspect_source()
    return built_mod.inspect_source()


# ---------------------------------------------------------------------------
# Schedule variants
# ---------------------------------------------------------------------------

def schedule_naive(M, N, K, tile=16):
    """Naive: one element per thread, no shared memory."""
    _, _, C = _matmul_compute(M, N, K)
    mod = tvm.IRModule.from_expr(te.create_prim_func([C.op.input_tensors[0], C.op.input_tensors[1], C]))
    sch = tir.Schedule(mod)
    block = sch.get_block("C")
    i, j, k = sch.get_loops(block)
    i0, i1 = sch.split(i, [None, tile])
    j0, j1 = sch.split(j, [None, tile])
    k0, k1 = sch.split(k, [None, tile])
    sch.reorder(i0, j0, k0, i1, j1, k1)
    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")
    sch.bind(i1, "threadIdx.y")
    sch.bind(j1, "threadIdx.x")
    return sch.mod


def schedule_optimized(M, N, K, bx=8, by=8, mx=4, my=4, mk=16):
    """Shared memory + register tiling + cooperative fetch + decomposed reduction."""
    A, B, C = _matmul_compute(M, N, K)
    mod = tvm.IRModule.from_expr(te.create_prim_func([A, B, C]))
    sch = tir.Schedule(mod)
    block = sch.get_block("C")
    i, j, k = sch.get_loops(block)

    # Multi-level tile: block -> thread -> register
    i_b, i_t, i_r = sch.split(i, [None, by, my])
    j_b, j_t, j_r = sch.split(j, [None, bx, mx])
    ko, ki = sch.split(k, [None, mk])
    sch.reorder(i_b, j_b, i_t, j_t, ko, ki, i_r, j_r)
    sch.bind(i_b, "blockIdx.y")
    sch.bind(j_b, "blockIdx.x")
    sch.bind(i_t, "threadIdx.y")
    sch.bind(j_t, "threadIdx.x")

    # Local accumulation
    c_local = sch.cache_write(block, 0, "local")
    sch.reverse_compute_at(c_local, j_t, preserve_unit_loops=True)

    # Shared memory cooperative fetch
    for idx in [0, 1]:
        shared = sch.cache_read(block, idx, "shared")
        n_loops = len(sch.get_loops(shared))
        sch.compute_at(shared, ko, preserve_unit_loops=True)
        loops = sch.get_loops(shared)[-n_loops:]
        fused = sch.fuse(*loops)
        ty, tx, remain = sch.split(fused, [by, bx, None])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    # Unroll + decompose
    sch.annotate(j_t, "pragma_auto_unroll_max_step", 256)
    sch.annotate(j_t, "pragma_unroll_explicit", 1)
    sch.decompose_reduction(block, ko)
    return sch.mod


def schedule_large_tile(M, N, K):
    """Larger tile: 16x16 threads, 4x4 register -> 64x64 output tile."""
    return schedule_optimized(M, N, K, bx=16, by=16, mx=4, my=4, mk=8)


def schedule_deep_k(M, N, K):
    """Deep K tile: larger reduction tile for better data reuse."""
    return schedule_optimized(M, N, K, bx=8, by=8, mx=4, my=4, mk=32)


def schedule_wide_register(M, N, K):
    """Wide register tile: 8x8 per thread, fewer threads."""
    return schedule_optimized(M, N, K, bx=4, by=4, mx=8, my=8, mk=16)


def schedule_tall_register(M, N, K):
    """Tall register: 8x2 per thread, more threads along x."""
    return schedule_optimized(M, N, K, bx=16, by=8, mx=2, my=8, mk=16)


VARIANTS = {
    "naive":         ("Naive: 1 elem/thread, no shared mem", schedule_naive),
    "optimized":     ("Shared mem + 4x4 register tile (8x8 threads)", schedule_optimized),
    "large_tile":    ("Shared mem + 4x4 register tile (16x16 threads)", schedule_large_tile),
    "deep_k":        ("Shared mem + deep K=32 reduction tile", schedule_deep_k),
    "wide_register": ("Shared mem + 8x8 register tile (4x4 threads)", schedule_wide_register),
    "tall_register": ("Shared mem + 8x2 register tile (16x8 threads)", schedule_tall_register),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(variant, M=1024, N=1024, K=1024):
    """Generate WGSL for a given variant."""
    desc, sched_fn = VARIANTS[variant]
    mod = sched_fn(M, N, K)
    target = tvm.target.Target("webgpu")
    built = tvm.build(mod, target=target)
    return _extract_wgsl(built), desc


def main():
    parser = argparse.ArgumentParser(description="Generate matmul FP32 WGSL shaders via TVM")
    parser.add_argument("--variant", choices=list(VARIANTS.keys()), default="optimized")
    parser.add_argument("--all", action="store_true", help="Generate all variants")
    parser.add_argument("--outdir", type=str, default=None, help="Write shaders to directory")
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--K", type=int, default=1024)
    args = parser.parse_args()

    variants = list(VARIANTS.keys()) if args.all else [args.variant]

    for name in variants:
        wgsl, desc = generate(name, args.M, args.N, args.K)
        header = f"// Variant: {name}\n// {desc}\n// Matrix: C[{args.M},{args.N}] = A[{args.M},{args.K}] @ B[{args.K},{args.N}]\n"

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
