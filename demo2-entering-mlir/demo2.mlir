/// @file
/// @author Alex Singer
/// @date   March 2025
/// @brief  The high-level code from demo 1 converted into MLIR.
///
/// Usually tools are used to automatically convert the user's code into MLIR
/// code. For this tutorial I wrote this kernel by hand, but it should be
/// trivial to build a tool to do this conversion for this particular example.

// This is used by the linalg.generic op later. See that section for more info.
// NOTE: affine_map<(i, j) -> (i, j)> means "index output[i,j] with the same
//       (i,j) as input" — i.e. no transposition, no broadcasting. Both ins
//       and outs use this same map, so it's a straight elementwise iteration.
#map = affine_map<(i, j) -> (i, j)>

// This is the top-level container operation. It is part of the "builtin" dialect.
//
//
// This operation contains all of the code which will be compiled through MLIR.
// It is used within the compiler to apply overall attributes to all operations
// within. For example, this is where the target device triple is stored.
module {

// This is the function operation as part of the "func" dialect. This dialect
// contains operations that have to do with defining and calling functions.
// FuncOps in MLIR have a name ("main" in this case), define arguments to the
// function, and define the output type. The function op contains one region
// containing ops which it will "execute" in order. The symantic of "executing"
// these ops in order comes from the abstraction of the FuncOp itself, not from
// MLIR's specifications.
// Here, we have a function named "main", it takes no arguments, and returns a
// tensor (which will be described later why).
func.func @main() -> tensor<256x1024xf32> {
    // This is the first occurence of a "Value" in this tutorial. Values are
    // what may get returned from operations. In MLIR, these are named using the
    // "%" symbol. Values in MLIR are Static-Single-Assignment (SSA). 

    // Values in MLIR always have a "type". In this case, this value is a Tensor
    // type. The Tensor type specifies a multi-dimensional array; however, there
    // is NO concept of memory (i.e. how the data is stored in the device). This
    // is by design. Tensors only represent a "chunk" of data, thats it. This is
    // A useful abstraction since it allows us to deal with compute at a higher
    // level of abstraction (not caring about how the buffers are allocated).
    //

    // NOTE: tensor.empty() allocates a tensor with UNDEFINED contents.
    //       No memory is actually allocated here; this is just a placeholder
    //       value at the tensor abstraction level. The compiler will decide
    //       how/when to allocate memory during bufferization.
    %FC_INPUT = tensor.empty() : tensor<256x512xf32>
    %FC_WEIGHT = tensor.empty() : tensor<512x1024xf32>

//============================================================================
    // Here we perform our first high-level linear-algebra operation. At a high
    // level, all we care about is that FC_INPUT and FC_WEIGHT are multiplied
    // together. 

    %c_init = arith.constant 0.0 : f32
    // NOTE: tensor.splat fills every element of the tensor with %c_init (0.0).
    //       Unlike tensor.empty(), the contents ARE defined — all zeros.
    //       This is required so the matmul MAC loop starts from 0.
    %matmul_init = tensor.splat %c_init : tensor<256x1024xf32>

    // NOTE: 'outs' is dual-purpose here:
    //   (1) INPUT role  — provides the initial value to accumulate into.
    //                     linalg.matmul expands to C[i,j] += A[i,k] * B[k,j],
    //                     so %matmul_init MUST be zero-initialized or the result
    //                     will be wrong.
    //   (2) OUTPUT role — its shape/type defines the result tensor.
    //                     The produced SSA value is bound to %FC_OUTPUT.
    // No memory is written at this level; bufferization handles that later.
    %FC_OUTPUT = linalg.matmul
                    ins(%FC_INPUT, %FC_WEIGHT : tensor<256x512xf32>, tensor<512x1024xf32>)
                    outs(%matmul_init : tensor<256x1024xf32>) -> tensor<256x1024xf32>

    // Our second high-level linear algebra operation that we wish to perform is
    // an elementwise ReLU operation. Currently, the linalg dialect does not
    // contain the ReLU activation function. MLIR generally contains ops for all
    // basic operations people may need, but ReLU may just not be common enough.
    // Luckily, in the linalg dialect there is a way to specify a "generic"
    // linear algebra operation.

    // Since the ReLU body overwrites every element unconditionally,
    // the initial contents of %relu_init don't matter — tensor.empty() is fine.
    // 'outs' here is only needed to declare the output shape/type.
    %relu_init = tensor.empty() : tensor<256x1024xf32>


    //There is one map per operand — one for each tensor in ins + outs. Here:
    //ins has 1 tensor (%FC_OUTPUT) → first #map
    //outs has 1 tensor (%relu_init) → second #map
    %OUT = linalg.generic { indexing_maps = [#map, #map],
                            iterator_types = ["parallel", "parallel"]}
               ins(%FC_OUTPUT : tensor<256x1024xf32>)
               outs(%relu_init : tensor<256x1024xf32>) {
               // Per-element body. For each (i, j):
               //   %in  = FC_OUTPUT[i, j]   (read-only, from ins)
               //   %out = relu_init[i, j]   (read-only initial value, from outs — ignored here)
               ^bb0(%in: f32, %out: f32):
                    %c0 = arith.constant 0.0 : f32
                    // NOTE: arith.cmpf ugt = floating-point "unordered greater than".
                    //       Returns i1: true if %in > 0.0, false otherwise.
                    //       "unordered" means NaN inputs yield true; use "ogt"
                    //       if you want NaN to yield false instead.
                    %cmp = arith.cmpf ugt, %in, %c0 : f32
                    // NOTE: arith.select is a ternary: if %cmp then %in else %c0.
                    //       Together with cmpf this implements: max(%in, 0.0) = ReLU.
                    %sel = arith.select %cmp, %in, %c0 : f32
                    // linalg.yield determines OUT[i, j] in the new output tensor.
                    // It does NOT write into %out; %out is just a block argument.
                    linalg.yield %sel : f32
               } -> tensor<256x1024xf32>

    // Return the final tensor result. This differs from the original main
    // function since MLIR is often smart enough to realize that this tensor is
    // never used and will optimize everything in this kernel away. To keep that
    // from happening for this tutorial, I just returned the result.
    func.return %OUT : tensor<256x1024xf32>
}

}