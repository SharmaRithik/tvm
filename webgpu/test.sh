#!/usr/bin/env bash
# Run all WebGPU and WASM tests.
set -euo pipefail

TVM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEB_DIR="$TVM_ROOT/web"
FAILED=0

echo "=== Node.js WASM tests ==="
cd "$WEB_DIR"
npx jest tests/node/ || FAILED=1

echo ""
echo "=== WebGPU pipeline tests (Python) ==="
cd "$TVM_ROOT"
python3 -m pytest tests/python/relax/test_pipeline.py -k "webgpu" -v || FAILED=1

echo ""
echo "=== WebGPU codegen smoke test ==="
python3 -c "
import tvm
from tvm import te
t = tvm.target.Target('webgpu')
n = te.var('n')
A = te.placeholder((n,), name='A')
B = te.compute(A.shape, lambda i: A[i] + 1.0, name='B')
mod = tvm.IRModule.from_expr(te.create_prim_func([A, B]))
sch = tvm.tir.Schedule(mod)
(i,) = sch.get_loops(block=sch.get_block('B'))
i0, i1 = sch.split(i, [None, 64])
sch.bind(i0, 'blockIdx.x')
sch.bind(i1, 'threadIdx.x')
f = tvm.build(sch.mod, target=t)
src = [m.inspect_source() for m in f.imports_ if m.kind == 'webgpu'][0]
assert 'workgroup_size' in src
print('WebGPU codegen: OK')
" || FAILED=1

if [ $FAILED -ne 0 ]; then
    echo ""
    echo "SOME TESTS FAILED"
    exit 1
fi

echo ""
echo "ALL TESTS PASSED"
