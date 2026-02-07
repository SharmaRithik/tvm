// Variant: naive
// Baseline: 1 elem/thread, no shared mem
// Matrix: C[1024,1024] = A[1024,1024] @ B[1024,1024]
//----------------------------------------
// Function: main_kernel
//----------------------------------------
@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read> B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;

struct PODArgs {
  packGridDimX: u32
}
@group(0) @binding(3) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(16, 16, 1)
fn main_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  for (var k : i32 = 0i; k < 1024i; k++) {
    if (k == 0i) {
      C[((((i32(blockIdx.y) * 16384i) + (i32(threadIdx.y) * 1024i)) + (v__1 * 16i)) + i32(threadIdx.x))] = 0.000000e+00f;
    }
    C[((((i32(blockIdx.y) * 16384i) + (i32(threadIdx.y) * 1024i)) + (v__1 * 16i)) + i32(threadIdx.x))] = fma(A[(((i32(blockIdx.y) * 16384i) + (i32(threadIdx.y) * 1024i)) + k)], B[(((k * 1024i) + (v__1 * 16i)) + i32(threadIdx.x))], C[((((i32(blockIdx.y) * 16384i) + (i32(threadIdx.y) * 1024i)) + (v__1 * 16i)) + i32(threadIdx.x))]);
  }
}

