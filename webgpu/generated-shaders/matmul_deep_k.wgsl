// Variant: deep_k
// Shared mem + deep K=32 reduction tile
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

var<workgroup> A_shared : array<f32, 1024>;
var<workgroup> B_shared : array<f32, 1024>;
@compute @workgroup_size(8, 8, 1)
fn main_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  var C_local : array<f32, 16>;
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  for (var var_1 : i32 = 0i; var_1 < 1i; var_1++) {
    C_local[0i] = 0.000000e+00f;
    C_local[1i] = 0.000000e+00f;
    C_local[2i] = 0.000000e+00f;
    C_local[3i] = 0.000000e+00f;
    C_local[4i] = 0.000000e+00f;
    C_local[5i] = 0.000000e+00f;
    C_local[6i] = 0.000000e+00f;
    C_local[7i] = 0.000000e+00f;
    C_local[8i] = 0.000000e+00f;
    C_local[9i] = 0.000000e+00f;
    C_local[10i] = 0.000000e+00f;
    C_local[11i] = 0.000000e+00f;
    C_local[12i] = 0.000000e+00f;
    C_local[13i] = 0.000000e+00f;
    C_local[14i] = 0.000000e+00f;
    C_local[15i] = 0.000000e+00f;
    for (var k_0 : i32 = 0i; k_0 < 32i; k_0++) {
      workgroupBarrier();
      A_shared[((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i))] = A[(((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (k_0 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i))];
      A_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 1i)] = A[((((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (k_0 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 1i)];
      A_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 2i)] = A[((((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (k_0 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 2i)];
      A_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 3i)] = A[((((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (k_0 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 3i)];
      A_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 4i)] = A[((((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (k_0 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 4i)];
      A_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 5i)] = A[((((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (k_0 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 5i)];
      A_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 6i)] = A[((((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (k_0 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 6i)];
      A_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 7i)] = A[((((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (k_0 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 7i)];
      A_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 8i)] = A[((((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (k_0 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 8i)];
      A_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 9i)] = A[((((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (k_0 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 9i)];
      A_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 10i)] = A[((((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (k_0 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 10i)];
      A_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 11i)] = A[((((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (k_0 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 11i)];
      A_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 12i)] = A[((((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (k_0 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 12i)];
      A_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 13i)] = A[((((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (k_0 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 13i)];
      A_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 14i)] = A[((((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (k_0 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 14i)];
      A_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 15i)] = A[((((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (k_0 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 15i)];
      B_shared[((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i))] = B[(((((k_0 * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (v__1 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i))];
      B_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 1i)] = B[((((((k_0 * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (v__1 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 1i)];
      B_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 2i)] = B[((((((k_0 * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (v__1 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 2i)];
      B_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 3i)] = B[((((((k_0 * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (v__1 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 3i)];
      B_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 4i)] = B[((((((k_0 * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (v__1 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 4i)];
      B_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 5i)] = B[((((((k_0 * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (v__1 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 5i)];
      B_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 6i)] = B[((((((k_0 * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (v__1 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 6i)];
      B_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 7i)] = B[((((((k_0 * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (v__1 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 7i)];
      B_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 8i)] = B[((((((k_0 * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (v__1 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 8i)];
      B_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 9i)] = B[((((((k_0 * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (v__1 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 9i)];
      B_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 10i)] = B[((((((k_0 * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (v__1 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 10i)];
      B_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 11i)] = B[((((((k_0 * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (v__1 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 11i)];
      B_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 12i)] = B[((((((k_0 * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (v__1 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 12i)];
      B_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 13i)] = B[((((((k_0 * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (v__1 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 13i)];
      B_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 14i)] = B[((((((k_0 * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (v__1 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 14i)];
      B_shared[(((i32(threadIdx.y) * 128i) + (i32(threadIdx.x) * 16i)) + 15i)] = B[((((((k_0 * 32768i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>1u) * 1024i)) + (v__1 * 32i)) + ((i32(threadIdx.x) & 1i) * 16i)) + 15i)];
      workgroupBarrier();
      for (var k_1 : i32 = 0i; k_1 < 32i; k_1++) {
        C_local[0i] = fma(A_shared[((i32(threadIdx.y) * 128i) + k_1)], B_shared[((k_1 * 32i) + (i32(threadIdx.x) * 4i))], C_local[0i]);
        C_local[1i] = fma(A_shared[((i32(threadIdx.y) * 128i) + k_1)], B_shared[(((k_1 * 32i) + (i32(threadIdx.x) * 4i)) + 1i)], C_local[1i]);
        C_local[2i] = fma(A_shared[((i32(threadIdx.y) * 128i) + k_1)], B_shared[(((k_1 * 32i) + (i32(threadIdx.x) * 4i)) + 2i)], C_local[2i]);
        C_local[3i] = fma(A_shared[((i32(threadIdx.y) * 128i) + k_1)], B_shared[(((k_1 * 32i) + (i32(threadIdx.x) * 4i)) + 3i)], C_local[3i]);
        C_local[4i] = fma(A_shared[(((i32(threadIdx.y) * 128i) + k_1) + 32i)], B_shared[((k_1 * 32i) + (i32(threadIdx.x) * 4i))], C_local[4i]);
        C_local[5i] = fma(A_shared[(((i32(threadIdx.y) * 128i) + k_1) + 32i)], B_shared[(((k_1 * 32i) + (i32(threadIdx.x) * 4i)) + 1i)], C_local[5i]);
        C_local[6i] = fma(A_shared[(((i32(threadIdx.y) * 128i) + k_1) + 32i)], B_shared[(((k_1 * 32i) + (i32(threadIdx.x) * 4i)) + 2i)], C_local[6i]);
        C_local[7i] = fma(A_shared[(((i32(threadIdx.y) * 128i) + k_1) + 32i)], B_shared[(((k_1 * 32i) + (i32(threadIdx.x) * 4i)) + 3i)], C_local[7i]);
        C_local[8i] = fma(A_shared[(((i32(threadIdx.y) * 128i) + k_1) + 64i)], B_shared[((k_1 * 32i) + (i32(threadIdx.x) * 4i))], C_local[8i]);
        C_local[9i] = fma(A_shared[(((i32(threadIdx.y) * 128i) + k_1) + 64i)], B_shared[(((k_1 * 32i) + (i32(threadIdx.x) * 4i)) + 1i)], C_local[9i]);
        C_local[10i] = fma(A_shared[(((i32(threadIdx.y) * 128i) + k_1) + 64i)], B_shared[(((k_1 * 32i) + (i32(threadIdx.x) * 4i)) + 2i)], C_local[10i]);
        C_local[11i] = fma(A_shared[(((i32(threadIdx.y) * 128i) + k_1) + 64i)], B_shared[(((k_1 * 32i) + (i32(threadIdx.x) * 4i)) + 3i)], C_local[11i]);
        C_local[12i] = fma(A_shared[(((i32(threadIdx.y) * 128i) + k_1) + 96i)], B_shared[((k_1 * 32i) + (i32(threadIdx.x) * 4i))], C_local[12i]);
        C_local[13i] = fma(A_shared[(((i32(threadIdx.y) * 128i) + k_1) + 96i)], B_shared[(((k_1 * 32i) + (i32(threadIdx.x) * 4i)) + 1i)], C_local[13i]);
        C_local[14i] = fma(A_shared[(((i32(threadIdx.y) * 128i) + k_1) + 96i)], B_shared[(((k_1 * 32i) + (i32(threadIdx.x) * 4i)) + 2i)], C_local[14i]);
        C_local[15i] = fma(A_shared[(((i32(threadIdx.y) * 128i) + k_1) + 96i)], B_shared[(((k_1 * 32i) + (i32(threadIdx.x) * 4i)) + 3i)], C_local[15i]);
      }
    }
    C[((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 32i)) + (i32(threadIdx.x) * 4i))] = C_local[0i];
    C[(((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 32i)) + (i32(threadIdx.x) * 4i)) + 1i)] = C_local[1i];
    C[(((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 32i)) + (i32(threadIdx.x) * 4i)) + 2i)] = C_local[2i];
    C[(((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 32i)) + (i32(threadIdx.x) * 4i)) + 3i)] = C_local[3i];
    C[(((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 32i)) + (i32(threadIdx.x) * 4i)) + 1024i)] = C_local[4i];
    C[(((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 32i)) + (i32(threadIdx.x) * 4i)) + 1025i)] = C_local[5i];
    C[(((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 32i)) + (i32(threadIdx.x) * 4i)) + 1026i)] = C_local[6i];
    C[(((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 32i)) + (i32(threadIdx.x) * 4i)) + 1027i)] = C_local[7i];
    C[(((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 32i)) + (i32(threadIdx.x) * 4i)) + 2048i)] = C_local[8i];
    C[(((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 32i)) + (i32(threadIdx.x) * 4i)) + 2049i)] = C_local[9i];
    C[(((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 32i)) + (i32(threadIdx.x) * 4i)) + 2050i)] = C_local[10i];
    C[(((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 32i)) + (i32(threadIdx.x) * 4i)) + 2051i)] = C_local[11i];
    C[(((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 32i)) + (i32(threadIdx.x) * 4i)) + 3072i)] = C_local[12i];
    C[(((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 32i)) + (i32(threadIdx.x) * 4i)) + 3073i)] = C_local[13i];
    C[(((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 32i)) + (i32(threadIdx.x) * 4i)) + 3074i)] = C_local[14i];
    C[(((((i32(blockIdx.y) * 32768i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 32i)) + (i32(threadIdx.x) * 4i)) + 3075i)] = C_local[15i];
  }
}

