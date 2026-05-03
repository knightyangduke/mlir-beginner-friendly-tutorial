module {
  func.func @main() -> memref<256x1024xf32> {
    %c512 = arith.constant 512 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x512xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<512x1024xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>

    // Zero-initialize %alloc_1 (the matmul output buffer).
    // This is the lowered form of `tensor.splat 0.0` from the original code.
    // Required because the matmul loop below accumulates (+=) into this buffer.
    //
    // C equivalent:
    //   for (int i = 0; i < 256; i++)
    //     for (int j = 0; j < 1024; j++)
    //       C[i][j] = 0.0f;
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        memref.store %cst, %alloc_1[%arg0, %arg1] : memref<256x1024xf32>
      }
    }

    // Matrix multiply: C = A * B  (linalg.matmul lowered to loops)
    //
    // C equivalent:
    //   for (int i = 0; i < 256; i++)
    //     for (int j = 0; j < 1024; j++)
    //       for (int k = 0; k < 512; k++)
    //         C[i][j] += A[i][k] * B[k][j];  // multiply-accumulate
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        scf.for %arg2 = %c0 to %c512 step %c1 {
          %0 = memref.load %alloc[%arg0, %arg2] : memref<256x512xf32>//A[i][k]
          %1 = memref.load %alloc_0[%arg2, %arg1] : memref<512x1024xf32>//B[k][j]
          %2 = memref.load %alloc_1[%arg0, %arg1] : memref<256x1024xf32>//C[i][j]
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          memref.store %4, %alloc_1[%arg0, %arg1] : memref<256x1024xf32>//C[i][j]
        }
      }
    }
    // ReLU: OUT = max(C[i][j], 0.0)  (linalg.generic lowered to loops)
    //
    // C equivalent:
    //   for (int i = 0; i < 256; i++)
    //     for (int j = 0; j < 1024; j++)
    //       OUT[i][j] = C[i][j] > 0.0f ? C[i][j] : 0.0f;
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>
    scf.for %arg0 = %c0 to %c256 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %0 = memref.load %alloc_1[%arg0, %arg1] : memref<256x1024xf32>
        %1 = arith.cmpf ugt, %0, %cst : f32
        %2 = arith.select %1, %0, %cst : f32
        memref.store %2, %alloc_2[%arg0, %arg1] : memref<256x1024xf32>
      }
    }
    return %alloc_2 : memref<256x1024xf32>
  }
}

