// Equivalent C (before loop fusion):
//   float A[256][512], B[512][1024];
//   float C[256][1024];  // 1 MB intermediate buffer
//   float out[256][1024];
//
//   // Loop nest 1: zero-initialize C
//   for (int i = 0; i < 256; i++)
//     for (int j = 0; j < 1024; j++)
//       C[i][j] = 0.0f;
//
//   // Loop nest 2: matmul into C
//   for (int i = 0; i < 256; i++)
//     for (int j = 0; j < 1024; j++)
//       for (int k = 0; k < 512; k++)
//         C[i][j] += A[i][k] * B[k][j];
//
//   // Loop nest 3: ReLU from C into out
//   for (int i = 0; i < 256; i++)
//     for (int j = 0; j < 1024; j++)
//       out[i][j] = C[i][j] > 0.0f ? C[i][j] : 0.0f;

// Equivalent C (after loop fusion):
//
//   float A[256][512], B[512][1024];
//   float out[256][1024];
//
//   // Single fused loop nest: matmul + ReLU per output element.
//   // The 1 MB intermediate buffer C is replaced by a single scalar 'acc'.
//   for (int i = 0; i < 256; i++) {
//     for (int j = 0; j < 1024; j++) {
//       float acc = 0.0f;                          // zero scalar, not full matrix
//       for (int k = 0; k < 512; k++)
//         acc += A[i][k] * B[k][j];               // matmul
//       out[i][j] = acc > 0.0f ? acc : 0.0f;      // ReLU immediately
//     }
//   }

module {
  func.func @main() -> memref<256x1024xf32> {
    %alloc = memref.alloc() : memref<1x1xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<256x512xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<512x1024xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>
    affine.for %arg0 = 0 to 256 {
      affine.for %arg1 = 0 to 1024 {
        affine.store %cst, %alloc[0, 0] : memref<1x1xf32>
        affine.for %arg2 = 0 to 512 {
          %3 = affine.load %alloc_0[%arg0, %arg2] : memref<256x512xf32>
          %4 = affine.load %alloc_1[%arg2, %arg1] : memref<512x1024xf32>
          %5 = affine.load %alloc[0, 0] : memref<1x1xf32>
          %6 = arith.mulf %3, %4 : f32
          %7 = arith.addf %5, %6 : f32
          affine.store %7, %alloc[0, 0] : memref<1x1xf32>
        }
        %0 = affine.load %alloc[0, 0] : memref<1x1xf32>
        %1 = arith.cmpf ugt, %0, %cst : f32
        %2 = arith.select %1, %0, %cst : f32
        affine.store %2, %alloc_2[%arg0, %arg1] : memref<256x1024xf32>
      }
    }
    return %alloc_2 : memref<256x1024xf32>
  }
}

