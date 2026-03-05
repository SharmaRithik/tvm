#!/usr/bin/env bash
set -euo pipefail

TVM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FAILED=0

echo "=== WebGPU codegen smoke test ==="
python3 -c "
import tvm
from tvm import te
from tvm.tir.build import split_host_device_mods, codegen_build
t = tvm.target.Target('webgpu')
h = tvm.target.Target({'kind': 'llvm', 'mtriple': 'wasm32-unknown-unknown-wasm'})
full = t.with_host(h)
A = te.placeholder((1024,), name='A')
B = te.compute(A.shape, lambda i: A[i] + 1.0, name='B')
mod = tvm.IRModule.from_expr(te.create_prim_func([A, B]))
sch = tvm.s_tir.Schedule(mod)
(i,) = sch.get_loops(block=sch.get_sblock('B'))
i0, i1 = sch.split(i, [None, 64])
sch.bind(i0, 'blockIdx.x')
sch.bind(i1, 'threadIdx.x')
with tvm.transform.PassContext(opt_level=3):
    m = tvm.tir.transform.BindTarget(full)(sch.mod)
    m = tvm.tir.get_tir_pipeline('default')(m)
    _, dd = split_host_device_mods(m)
    for tgt, dm in dd.items():
        dm = tvm.tir.pipeline.finalize_device_passes()(dm)
        src = codegen_build(dm, tgt).inspect_source('')
        assert 'workgroup_size' in src
        print('codegen: OK')
" || FAILED=1

echo "=== Shader generation test ==="
cd "$TVM_ROOT"
python3 webgpu/generate_shaders.py --size 256 --outdir /tmp/webgpu_test_shaders || FAILED=1

if [ $FAILED -ne 0 ]; then
    echo "SOME TESTS FAILED"
    exit 1
fi

echo "ALL TESTS PASSED"
