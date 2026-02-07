// Variant: large_tile
// Shared mem + 4x4 register tile (16x16 threads)
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

var<workgroup> A_shared : array<f32, 512>;
var<workgroup> B_shared : array<f32, 512>;
@compute @workgroup_size(16, 16, 1)
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
    for (var k_0 : i32 = 0i; k_0 < 128i; k_0++) {
      workgroupBarrier();
      A_shared[((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 2i))] = A[(((((i32(blockIdx.y) * 65536i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>2u) * 1024i)) + (k_0 * 8i)) + ((i32(threadIdx.x) & 3i) * 2i))];
      A_shared[(((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 2i)) + 1i)] = A[((((((i32(blockIdx.y) * 65536i) + (i32(threadIdx.y) * 4096i)) + ((i32(threadIdx.x)>>2u) * 1024i)) + (k_0 * 8i)) + ((i32(threadIdx.x) & 3i) * 2i)) + 1i)];
      B_shared[((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 2i))] = B[(((((k_0 * 8192i) + ((i32(threadIdx.y)>>1u) * 1024i)) + (v__1 * 64i)) + ((i32(threadIdx.y) & 1i) * 32i)) + (i32(threadIdx.x) * 2i))];
      B_shared[(((i32(threadIdx.y)>>1u) * 64i) + ((((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 2i)) + 1i) & 63i))] = B[((((k_0 * 8192i) + ((i32(threadIdx.y)>>1u) * 1024i)) + (v__1 * 64i)) + ((((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 2i)) + 1i) & 63i))];
      workgroupBarrier();
      C_local[0i] = fma(A_shared[(i32(threadIdx.y) * 32i)], B_shared[(i32(threadIdx.x) * 4i)], C_local[0i]);
      C_local[1i] = fma(A_shared[(i32(threadIdx.y) * 32i)], B_shared[((i32(threadIdx.x) * 4i) + 1i)], C_local[1i]);
      C_local[2i] = fma(A_shared[(i32(threadIdx.y) * 32i)], B_shared[((i32(threadIdx.x) * 4i) + 2i)], C_local[2i]);
      C_local[3i] = fma(A_shared[(i32(threadIdx.y) * 32i)], B_shared[((i32(threadIdx.x) * 4i) + 3i)], C_local[3i]);
      C_local[4i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 8i)], B_shared[(i32(threadIdx.x) * 4i)], C_local[4i]);
      C_local[5i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 8i)], B_shared[((i32(threadIdx.x) * 4i) + 1i)], C_local[5i]);
      C_local[6i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 8i)], B_shared[((i32(threadIdx.x) * 4i) + 2i)], C_local[6i]);
      C_local[7i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 8i)], B_shared[((i32(threadIdx.x) * 4i) + 3i)], C_local[7i]);
      C_local[8i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 16i)], B_shared[(i32(threadIdx.x) * 4i)], C_local[8i]);
      C_local[9i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 16i)], B_shared[((i32(threadIdx.x) * 4i) + 1i)], C_local[9i]);
      C_local[10i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 16i)], B_shared[((i32(threadIdx.x) * 4i) + 2i)], C_local[10i]);
      C_local[11i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 16i)], B_shared[((i32(threadIdx.x) * 4i) + 3i)], C_local[11i]);
      C_local[12i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 24i)], B_shared[(i32(threadIdx.x) * 4i)], C_local[12i]);
      C_local[13i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 24i)], B_shared[((i32(threadIdx.x) * 4i) + 1i)], C_local[13i]);
      C_local[14i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 24i)], B_shared[((i32(threadIdx.x) * 4i) + 2i)], C_local[14i]);
      C_local[15i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 24i)], B_shared[((i32(threadIdx.x) * 4i) + 3i)], C_local[15i]);
      C_local[0i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 1i)], B_shared[((i32(threadIdx.x) * 4i) + 64i)], C_local[0i]);
      C_local[1i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 1i)], B_shared[((i32(threadIdx.x) * 4i) + 65i)], C_local[1i]);
      C_local[2i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 1i)], B_shared[((i32(threadIdx.x) * 4i) + 66i)], C_local[2i]);
      C_local[3i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 1i)], B_shared[((i32(threadIdx.x) * 4i) + 67i)], C_local[3i]);
      C_local[4i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 9i)], B_shared[((i32(threadIdx.x) * 4i) + 64i)], C_local[4i]);
      C_local[5i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 9i)], B_shared[((i32(threadIdx.x) * 4i) + 65i)], C_local[5i]);
      C_local[6i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 9i)], B_shared[((i32(threadIdx.x) * 4i) + 66i)], C_local[6i]);
      C_local[7i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 9i)], B_shared[((i32(threadIdx.x) * 4i) + 67i)], C_local[7i]);
      C_local[8i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 17i)], B_shared[((i32(threadIdx.x) * 4i) + 64i)], C_local[8i]);
      C_local[9i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 17i)], B_shared[((i32(threadIdx.x) * 4i) + 65i)], C_local[9i]);
      C_local[10i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 17i)], B_shared[((i32(threadIdx.x) * 4i) + 66i)], C_local[10i]);
      C_local[11i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 17i)], B_shared[((i32(threadIdx.x) * 4i) + 67i)], C_local[11i]);
      C_local[12i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 25i)], B_shared[((i32(threadIdx.x) * 4i) + 64i)], C_local[12i]);
      C_local[13i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 25i)], B_shared[((i32(threadIdx.x) * 4i) + 65i)], C_local[13i]);
      C_local[14i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 25i)], B_shared[((i32(threadIdx.x) * 4i) + 66i)], C_local[14i]);
      C_local[15i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 25i)], B_shared[((i32(threadIdx.x) * 4i) + 67i)], C_local[15i]);
      C_local[0i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 2i)], B_shared[((i32(threadIdx.x) * 4i) + 128i)], C_local[0i]);
      C_local[1i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 2i)], B_shared[((i32(threadIdx.x) * 4i) + 129i)], C_local[1i]);
      C_local[2i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 2i)], B_shared[((i32(threadIdx.x) * 4i) + 130i)], C_local[2i]);
      C_local[3i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 2i)], B_shared[((i32(threadIdx.x) * 4i) + 131i)], C_local[3i]);
      C_local[4i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 10i)], B_shared[((i32(threadIdx.x) * 4i) + 128i)], C_local[4i]);
      C_local[5i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 10i)], B_shared[((i32(threadIdx.x) * 4i) + 129i)], C_local[5i]);
      C_local[6i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 10i)], B_shared[((i32(threadIdx.x) * 4i) + 130i)], C_local[6i]);
      C_local[7i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 10i)], B_shared[((i32(threadIdx.x) * 4i) + 131i)], C_local[7i]);
      C_local[8i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 18i)], B_shared[((i32(threadIdx.x) * 4i) + 128i)], C_local[8i]);
      C_local[9i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 18i)], B_shared[((i32(threadIdx.x) * 4i) + 129i)], C_local[9i]);
      C_local[10i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 18i)], B_shared[((i32(threadIdx.x) * 4i) + 130i)], C_local[10i]);
      C_local[11i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 18i)], B_shared[((i32(threadIdx.x) * 4i) + 131i)], C_local[11i]);
      C_local[12i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 26i)], B_shared[((i32(threadIdx.x) * 4i) + 128i)], C_local[12i]);
      C_local[13i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 26i)], B_shared[((i32(threadIdx.x) * 4i) + 129i)], C_local[13i]);
      C_local[14i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 26i)], B_shared[((i32(threadIdx.x) * 4i) + 130i)], C_local[14i]);
      C_local[15i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 26i)], B_shared[((i32(threadIdx.x) * 4i) + 131i)], C_local[15i]);
      C_local[0i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 3i)], B_shared[((i32(threadIdx.x) * 4i) + 192i)], C_local[0i]);
      C_local[1i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 3i)], B_shared[((i32(threadIdx.x) * 4i) + 193i)], C_local[1i]);
      C_local[2i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 3i)], B_shared[((i32(threadIdx.x) * 4i) + 194i)], C_local[2i]);
      C_local[3i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 3i)], B_shared[((i32(threadIdx.x) * 4i) + 195i)], C_local[3i]);
      C_local[4i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 11i)], B_shared[((i32(threadIdx.x) * 4i) + 192i)], C_local[4i]);
      C_local[5i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 11i)], B_shared[((i32(threadIdx.x) * 4i) + 193i)], C_local[5i]);
      C_local[6i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 11i)], B_shared[((i32(threadIdx.x) * 4i) + 194i)], C_local[6i]);
      C_local[7i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 11i)], B_shared[((i32(threadIdx.x) * 4i) + 195i)], C_local[7i]);
      C_local[8i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 19i)], B_shared[((i32(threadIdx.x) * 4i) + 192i)], C_local[8i]);
      C_local[9i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 19i)], B_shared[((i32(threadIdx.x) * 4i) + 193i)], C_local[9i]);
      C_local[10i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 19i)], B_shared[((i32(threadIdx.x) * 4i) + 194i)], C_local[10i]);
      C_local[11i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 19i)], B_shared[((i32(threadIdx.x) * 4i) + 195i)], C_local[11i]);
      C_local[12i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 27i)], B_shared[((i32(threadIdx.x) * 4i) + 192i)], C_local[12i]);
      C_local[13i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 27i)], B_shared[((i32(threadIdx.x) * 4i) + 193i)], C_local[13i]);
      C_local[14i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 27i)], B_shared[((i32(threadIdx.x) * 4i) + 194i)], C_local[14i]);
      C_local[15i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 27i)], B_shared[((i32(threadIdx.x) * 4i) + 195i)], C_local[15i]);
      C_local[0i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 4i)], B_shared[((i32(threadIdx.x) * 4i) + 256i)], C_local[0i]);
      C_local[1i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 4i)], B_shared[((i32(threadIdx.x) * 4i) + 257i)], C_local[1i]);
      C_local[2i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 4i)], B_shared[((i32(threadIdx.x) * 4i) + 258i)], C_local[2i]);
      C_local[3i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 4i)], B_shared[((i32(threadIdx.x) * 4i) + 259i)], C_local[3i]);
      C_local[4i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 12i)], B_shared[((i32(threadIdx.x) * 4i) + 256i)], C_local[4i]);
      C_local[5i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 12i)], B_shared[((i32(threadIdx.x) * 4i) + 257i)], C_local[5i]);
      C_local[6i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 12i)], B_shared[((i32(threadIdx.x) * 4i) + 258i)], C_local[6i]);
      C_local[7i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 12i)], B_shared[((i32(threadIdx.x) * 4i) + 259i)], C_local[7i]);
      C_local[8i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 20i)], B_shared[((i32(threadIdx.x) * 4i) + 256i)], C_local[8i]);
      C_local[9i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 20i)], B_shared[((i32(threadIdx.x) * 4i) + 257i)], C_local[9i]);
      C_local[10i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 20i)], B_shared[((i32(threadIdx.x) * 4i) + 258i)], C_local[10i]);
      C_local[11i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 20i)], B_shared[((i32(threadIdx.x) * 4i) + 259i)], C_local[11i]);
      C_local[12i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 28i)], B_shared[((i32(threadIdx.x) * 4i) + 256i)], C_local[12i]);
      C_local[13i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 28i)], B_shared[((i32(threadIdx.x) * 4i) + 257i)], C_local[13i]);
      C_local[14i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 28i)], B_shared[((i32(threadIdx.x) * 4i) + 258i)], C_local[14i]);
      C_local[15i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 28i)], B_shared[((i32(threadIdx.x) * 4i) + 259i)], C_local[15i]);
      C_local[0i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 5i)], B_shared[((i32(threadIdx.x) * 4i) + 320i)], C_local[0i]);
      C_local[1i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 5i)], B_shared[((i32(threadIdx.x) * 4i) + 321i)], C_local[1i]);
      C_local[2i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 5i)], B_shared[((i32(threadIdx.x) * 4i) + 322i)], C_local[2i]);
      C_local[3i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 5i)], B_shared[((i32(threadIdx.x) * 4i) + 323i)], C_local[3i]);
      C_local[4i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 13i)], B_shared[((i32(threadIdx.x) * 4i) + 320i)], C_local[4i]);
      C_local[5i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 13i)], B_shared[((i32(threadIdx.x) * 4i) + 321i)], C_local[5i]);
      C_local[6i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 13i)], B_shared[((i32(threadIdx.x) * 4i) + 322i)], C_local[6i]);
      C_local[7i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 13i)], B_shared[((i32(threadIdx.x) * 4i) + 323i)], C_local[7i]);
      C_local[8i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 21i)], B_shared[((i32(threadIdx.x) * 4i) + 320i)], C_local[8i]);
      C_local[9i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 21i)], B_shared[((i32(threadIdx.x) * 4i) + 321i)], C_local[9i]);
      C_local[10i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 21i)], B_shared[((i32(threadIdx.x) * 4i) + 322i)], C_local[10i]);
      C_local[11i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 21i)], B_shared[((i32(threadIdx.x) * 4i) + 323i)], C_local[11i]);
      C_local[12i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 29i)], B_shared[((i32(threadIdx.x) * 4i) + 320i)], C_local[12i]);
      C_local[13i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 29i)], B_shared[((i32(threadIdx.x) * 4i) + 321i)], C_local[13i]);
      C_local[14i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 29i)], B_shared[((i32(threadIdx.x) * 4i) + 322i)], C_local[14i]);
      C_local[15i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 29i)], B_shared[((i32(threadIdx.x) * 4i) + 323i)], C_local[15i]);
      C_local[0i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 6i)], B_shared[((i32(threadIdx.x) * 4i) + 384i)], C_local[0i]);
      C_local[1i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 6i)], B_shared[((i32(threadIdx.x) * 4i) + 385i)], C_local[1i]);
      C_local[2i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 6i)], B_shared[((i32(threadIdx.x) * 4i) + 386i)], C_local[2i]);
      C_local[3i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 6i)], B_shared[((i32(threadIdx.x) * 4i) + 387i)], C_local[3i]);
      C_local[4i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 14i)], B_shared[((i32(threadIdx.x) * 4i) + 384i)], C_local[4i]);
      C_local[5i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 14i)], B_shared[((i32(threadIdx.x) * 4i) + 385i)], C_local[5i]);
      C_local[6i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 14i)], B_shared[((i32(threadIdx.x) * 4i) + 386i)], C_local[6i]);
      C_local[7i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 14i)], B_shared[((i32(threadIdx.x) * 4i) + 387i)], C_local[7i]);
      C_local[8i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 22i)], B_shared[((i32(threadIdx.x) * 4i) + 384i)], C_local[8i]);
      C_local[9i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 22i)], B_shared[((i32(threadIdx.x) * 4i) + 385i)], C_local[9i]);
      C_local[10i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 22i)], B_shared[((i32(threadIdx.x) * 4i) + 386i)], C_local[10i]);
      C_local[11i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 22i)], B_shared[((i32(threadIdx.x) * 4i) + 387i)], C_local[11i]);
      C_local[12i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 30i)], B_shared[((i32(threadIdx.x) * 4i) + 384i)], C_local[12i]);
      C_local[13i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 30i)], B_shared[((i32(threadIdx.x) * 4i) + 385i)], C_local[13i]);
      C_local[14i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 30i)], B_shared[((i32(threadIdx.x) * 4i) + 386i)], C_local[14i]);
      C_local[15i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 30i)], B_shared[((i32(threadIdx.x) * 4i) + 387i)], C_local[15i]);
      C_local[0i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 7i)], B_shared[((i32(threadIdx.x) * 4i) + 448i)], C_local[0i]);
      C_local[1i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 7i)], B_shared[((i32(threadIdx.x) * 4i) + 449i)], C_local[1i]);
      C_local[2i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 7i)], B_shared[((i32(threadIdx.x) * 4i) + 450i)], C_local[2i]);
      C_local[3i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 7i)], B_shared[((i32(threadIdx.x) * 4i) + 451i)], C_local[3i]);
      C_local[4i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 15i)], B_shared[((i32(threadIdx.x) * 4i) + 448i)], C_local[4i]);
      C_local[5i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 15i)], B_shared[((i32(threadIdx.x) * 4i) + 449i)], C_local[5i]);
      C_local[6i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 15i)], B_shared[((i32(threadIdx.x) * 4i) + 450i)], C_local[6i]);
      C_local[7i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 15i)], B_shared[((i32(threadIdx.x) * 4i) + 451i)], C_local[7i]);
      C_local[8i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 23i)], B_shared[((i32(threadIdx.x) * 4i) + 448i)], C_local[8i]);
      C_local[9i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 23i)], B_shared[((i32(threadIdx.x) * 4i) + 449i)], C_local[9i]);
      C_local[10i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 23i)], B_shared[((i32(threadIdx.x) * 4i) + 450i)], C_local[10i]);
      C_local[11i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 23i)], B_shared[((i32(threadIdx.x) * 4i) + 451i)], C_local[11i]);
      C_local[12i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 31i)], B_shared[((i32(threadIdx.x) * 4i) + 448i)], C_local[12i]);
      C_local[13i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 31i)], B_shared[((i32(threadIdx.x) * 4i) + 449i)], C_local[13i]);
      C_local[14i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 31i)], B_shared[((i32(threadIdx.x) * 4i) + 450i)], C_local[14i]);
      C_local[15i] = fma(A_shared[((i32(threadIdx.y) * 32i) + 31i)], B_shared[((i32(threadIdx.x) * 4i) + 451i)], C_local[15i]);
    }
    C[((((i32(blockIdx.y) * 65536i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 64i)) + (i32(threadIdx.x) * 4i))] = C_local[0i];
    C[(((((i32(blockIdx.y) * 65536i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 64i)) + (i32(threadIdx.x) * 4i)) + 1i)] = C_local[1i];
    C[(((((i32(blockIdx.y) * 65536i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 64i)) + (i32(threadIdx.x) * 4i)) + 2i)] = C_local[2i];
    C[(((((i32(blockIdx.y) * 65536i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 64i)) + (i32(threadIdx.x) * 4i)) + 3i)] = C_local[3i];
    C[(((((i32(blockIdx.y) * 65536i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 64i)) + (i32(threadIdx.x) * 4i)) + 1024i)] = C_local[4i];
    C[(((((i32(blockIdx.y) * 65536i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 64i)) + (i32(threadIdx.x) * 4i)) + 1025i)] = C_local[5i];
    C[(((((i32(blockIdx.y) * 65536i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 64i)) + (i32(threadIdx.x) * 4i)) + 1026i)] = C_local[6i];
    C[(((((i32(blockIdx.y) * 65536i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 64i)) + (i32(threadIdx.x) * 4i)) + 1027i)] = C_local[7i];
    C[(((((i32(blockIdx.y) * 65536i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 64i)) + (i32(threadIdx.x) * 4i)) + 2048i)] = C_local[8i];
    C[(((((i32(blockIdx.y) * 65536i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 64i)) + (i32(threadIdx.x) * 4i)) + 2049i)] = C_local[9i];
    C[(((((i32(blockIdx.y) * 65536i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 64i)) + (i32(threadIdx.x) * 4i)) + 2050i)] = C_local[10i];
    C[(((((i32(blockIdx.y) * 65536i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 64i)) + (i32(threadIdx.x) * 4i)) + 2051i)] = C_local[11i];
    C[(((((i32(blockIdx.y) * 65536i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 64i)) + (i32(threadIdx.x) * 4i)) + 3072i)] = C_local[12i];
    C[(((((i32(blockIdx.y) * 65536i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 64i)) + (i32(threadIdx.x) * 4i)) + 3073i)] = C_local[13i];
    C[(((((i32(blockIdx.y) * 65536i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 64i)) + (i32(threadIdx.x) * 4i)) + 3074i)] = C_local[14i];
    C[(((((i32(blockIdx.y) * 65536i) + (i32(threadIdx.y) * 4096i)) + (v__1 * 64i)) + (i32(threadIdx.x) * 4i)) + 3075i)] = C_local[15i];
  }
}

