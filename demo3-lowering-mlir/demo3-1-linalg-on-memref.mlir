#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main() -> memref<256x1024xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x512xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<512x1024xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>
    // Initialize %alloc_1 to zero — the memref equivalent of:
    //   %splat = tensor.splat %cst : tensor<256x1024xf32>
    // In the tensor world, splat produced a new SSA value used as the
    // matmul accumulator. Here, bufferization allocated %alloc_1 first (see line 7 above) and
    // then fills it in-place with linalg.map (no ins, just a scalar yield).
    linalg.map outs(%alloc_1 : memref<256x1024xf32>)
      () {
        linalg.yield %cst : f32
      }
    // This is the *same* linalg.matmul op as in the tensor case:
    //   %2 = linalg.matmul ins(%0, %1 : tensor<...>) outs(%splat : tensor<...>) -> tensor<256x1024xf32>
    // The op definition accepts both tensor and memref via AnyShaped constraints.
    // The key difference is driven purely by types (DestinationStyleOpInterface):
    //   - tensor outs  → produces a new SSA result (value semantics, immutable)
    //   - memref outs  → no result; writes in-place through the buffer (buffer semantics)
    // Bufferization replaced tensor operands with memrefs, which automatically
    // drops the return value — no separate op needed, the same opcode covers both worlds.
    linalg.matmul ins(%alloc, %alloc_0 : memref<256x512xf32>, memref<512x1024xf32>) outs(%alloc_1 : memref<256x1024xf32>)
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<256x1024xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_1 : memref<256x1024xf32>) outs(%alloc_2 : memref<256x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.cmpf ugt, %in, %cst : f32
      %1 = arith.select %0, %in, %cst : f32
      linalg.yield %1 : f32
    }
    return %alloc_2 : memref<256x1024xf32>
  }
}

