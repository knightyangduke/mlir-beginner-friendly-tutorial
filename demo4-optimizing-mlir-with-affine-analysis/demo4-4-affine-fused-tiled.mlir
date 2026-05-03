// Equivalent C (before tiling, i.e. after loop fusion):
//
//   float A[256][512], B[512][1024];
//   float out[256][1024];
//
//   for (int i = 0; i < 256; i++) {
//     for (int j = 0; j < 1024; j++) {
//       float acc = 0.0f;
//       for (int k = 0; k < 512; k++)
//         acc += A[i][k] * B[k][j];
//       out[i][j] = acc > 0.0f ? acc : 0.0f;
//     }
//   }

// Equivalent C (after tiling, tile sizes TI=2, TJ=2):
//
//   float A[256][512], B[512][1024];
//   float out[256][1024];
//
//   // Outer tile loops iterate over 2x2 blocks of output elements.
//   // Each tile's working set: A[2][512] + B[512][2] = 8 KB, fits in L2 cache.
//   for (int ii = 0; ii < 256; ii += 2) {       // 128 tiles along i
//     for (int jj = 0; jj < 1024; jj += 2) {    // 512 tiles along j
//       for (int i = ii; i < ii + 2; i++) {      // inner i (within tile)
//         for (int j = jj; j < jj + 2; j++) {   // inner j (within tile)
//           float acc = 0.0f;
//           for (int k = 0; k < 512; k++)        // A[i][*] and B[*][j] stay in cache
//             acc += A[i][k] * B[k][j];
//           out[i][j] = acc > 0.0f ? acc : 0.0f;
//         }
//       }
//     }
//   }

module {
  func.func @main() -> memref<256x1024xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<1x1xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<256x512xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<512x1024xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>
    affine.for %arg0 = 0 to 128 {
      affine.for %arg1 = 0 to 512 {
        affine.for %arg2 = 0 to 2 {
          affine.for %arg3 = 0 to 2 {
            affine.store %cst, %alloc[0, 0] : memref<1x1xf32>
            affine.for %arg4 = 0 to 512 {
              %3 = affine.load %alloc_0[%arg2 + %arg0 * 2, %arg4] : memref<256x512xf32>
              %4 = affine.load %alloc_1[%arg4, %arg3 + %arg1 * 2] : memref<512x1024xf32>
              %5 = affine.load %alloc[0, 0] : memref<1x1xf32>
              %6 = arith.mulf %3, %4 : f32
              %7 = arith.addf %5, %6 : f32
              affine.store %7, %alloc[0, 0] : memref<1x1xf32>
            }
            %0 = affine.load %alloc[0, 0] : memref<1x1xf32>
            %1 = arith.cmpf ugt, %0, %cst : f32
            %2 = arith.select %1, %0, %cst : f32
            affine.store %2, %alloc_2[%arg2 + %arg0 * 2, %arg3 + %arg1 * 2] : memref<256x1024xf32>
          }
        }
      }
    }
    return %alloc_2 : memref<256x1024xf32>
  }
}

