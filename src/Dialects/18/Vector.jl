module vector

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`vscale`

The `vscale` op returns the scale of the scalable vectors, a positive
integer value that is constant at runtime but unknown at compile-time.
The scale of the vector indicates the multiplicity of the vectors and
vector operations. For example, a `vector<[4]xi32>` is equivalent to
`vscale` consecutive `vector<4xi32>`; and an operation on a
`vector<[4]xi32>` is equivalent to performing that operation `vscale`
times, once on each `<4xi32>` segment of the scalable vector. The `vscale`
op can be used to calculate the step in vector-length agnostic (VLA) loops.
Right now we only support one contiguous set of scalable dimensions, all of
them grouped and scaled with the value returned by \'vscale\'.
"""
function vscale(; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "vector.vscale", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`bitcast`

The bitcast operation casts between vectors of the same rank, the minor 1-D
vector size is casted to a vector with a different element type but same
bitwidth. In case of 0-D vectors, the bitwidth of element types must be
equal.

# Example

```mlir
// Example casting to a smaller element type.
%1 = vector.bitcast %0 : vector<5x1x4x3xf32> to vector<5x1x4x6xi16>

// Example casting to a bigger element type.
%3 = vector.bitcast %2 : vector<10x12x8xi8> to vector<10x12x2xi32>

// Example casting to an element type of the same size.
%5 = vector.bitcast %4 : vector<5x1x4x3xf32> to vector<5x1x4x3xi32>

// Example casting of 0-D vectors.
%7 = vector.bitcast %6 : vector<f32> to vector<i32>
```
"""
function bitcast(source::Value; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[source, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "vector.bitcast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`broadcast`

Broadcasts the scalar or k-D vector value in the source operand
to a n-D result vector such that the broadcast makes sense, i.e.,
the source operand is duplicated to match the given rank and sizes
in the result vector. The legality rules are:
* the source operand must have the same element type as the result type
* a k-D vector <s_1 x .. x s_k x type> can be broadcast to
  a n-D vector <t_1 x .. x t_n x type> if
   * k <= n, and
   * the sizes in the trailing dimensions n-k < i <= n with j=i+k-n
      match exactly as s_j = t_i or s_j = 1:
   ```
       t_1 x   ..  t_n-k x t_n-k+1 x .. x t_i x .. x t_n
                           s_1     x .. x s_j x .. x s_k
           <duplication>         <potential stretch>
   ```
The source operand is duplicated over all the missing leading dimensions
and stretched over the trailing dimensions where the source has a non-equal
dimension of 1. These rules imply that any scalar broadcast (k=0) to any
shaped vector with the same element type is always legal.

# Example

```mlir
%0 = arith.constant 0.0 : f32
%1 = vector.broadcast %0 : f32 to vector<16xf32>
%2 = vector.broadcast %1 : vector<16xf32> to vector<4x16xf32>
```
"""
function broadcast(source::Value; vector::IR.Type, location=Location())
    results = IR.Type[vector, ]
    operands = Value[source, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "vector.broadcast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`compressstore`

The compress store operation writes elements from a 1-D vector into memory
as defined by a base with indices and a 1-D mask vector. When the mask is
set, the corresponding element from the vector is written next to memory.
Otherwise, no action is taken for the element. Informally the semantics are:
```
index = i
if (mask[0]) base[index++] = value[0]
if (mask[1]) base[index++] = value[1]
etc.
```
Note that the index increment is done conditionally.

If a mask bit is set and the corresponding index is out-of-bounds for the
given base, the behavior is undefined. If a mask bit is not set, no value
is stored regardless of the index, and the index is allowed to be
out-of-bounds.

The compress store can be used directly where applicable, or can be used
during progressively lowering to bring other memory operations closer to
hardware ISA support for a compress. The semantics of the operation closely
correspond to those of the `llvm.masked.compressstore`
[intrinsic](https://llvm.org/docs/LangRef.html#llvm-masked-compressstore-intrinsics).

Examples:

```mlir
vector.compressstore %base[%i], %mask, %value
  : memref<?xf32>, vector<8xi1>, vector<8xf32>

vector.compressstore %base[%i, %j], %mask, %value
  : memref<?x?xf32>, vector<16xi1>, vector<16xf32>
```
"""
function compressstore(base::Value, indices::Vector{Value}, mask::Value, valueToStore::Value; location=Location())
    results = IR.Type[]
    operands = Value[base, indices..., mask, valueToStore, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "vector.compressstore", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`constant_mask`

Creates and returns a vector mask where elements of the result vector
are set to \'0\' or \'1\', based on whether the element indices are contained
within a hyper-rectangular region specified by the \'mask_dim_sizes\'
array attribute argument. Each element of the \'mask_dim_sizes\' array,
specifies an exclusive upper bound [0, mask-dim-size-element-value)
for a unique dimension in the vector result. The conjunction of the ranges
define a hyper-rectangular region within which elements values are set to 1
(otherwise element values are set to 0). Each value of \'mask_dim_sizes\' must
be non-negative and not greater than the size of the corresponding vector
dimension (as opposed to vector.create_mask which allows this). Sizes that
correspond to scalable dimensions are implicitly multiplied by vscale,
though currently only zero (none set) or the size of the dim/vscale
(all set) are supported.

# Example

```mlir
// create a constant vector mask of size 4x3xi1 with elements in range
// 0 <= row <= 2 and 0 <= col <= 1 are set to 1 (others to 0).
%1 = vector.constant_mask [3, 2] : vector<4x3xi1>

print %1
              columns
            0    1    2
          |------------
        0 | 1    1    0
  rows  1 | 1    1    0
        2 | 1    1    0
        3 | 0    0    0
```
"""
function constant_mask(; result_0::IR.Type, mask_dim_sizes, location=Location())
    results = IR.Type[result_0, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("mask_dim_sizes", mask_dim_sizes), ]
    
    create_operation(
        "vector.constant_mask", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`contract`

Computes the sum of products of vector elements along contracting
dimension pairs from 2 vectors of rank M and N respectively, adds this
intermediate result to the accumulator argument of rank K, and returns a
vector result of rank K (where K = num_lhs_free_dims + num_rhs_free_dims +
num_batch_dims (see dimension type descriptions below)). For K = 0 (no
free or batch dimensions), the accumulator and output are a scalar.

If operands and the result have types of different bitwidths, operands are
promoted to have the same bitwidth as the result before performing the
contraction. For integer types, only signless integer types are supported,
and the promotion happens via sign extension.

An iterator type attribute list must be specified, where each element of
the list represents an iterator with one of the following types:

*   \"reduction\": reduction dimensions are present in the lhs and rhs
    arguments but not in the output (and accumulator
    argument). These are the dimensions along which the vector
    contraction op computes the sum of products, and
    contracting dimension pair dimension sizes must match
    between lhs/rhs.

*   \"parallel\": Batch dimensions are iterator type \"parallel\", and
    are non-contracting dimensions present in the lhs, rhs and
    output. The lhs/rhs co-iterate along the batch dimensions,
    which should be expressed in their indexing maps.

    Free dimensions are iterator type \"parallel\", and are
    non-contraction, non-batch dimensions accessed by either the
    lhs or rhs (but not both). The lhs and rhs free dimensions
    are unrelated to each other and do not co-iterate, which
    should be expressed in their indexing maps.

An indexing map attribute list must be specified with an entry for lhs, rhs
and acc arguments. An indexing map attribute specifies a mapping from each
iterator in the iterator type list, to each dimension of an N-D vector.

An optional kind attribute may be used to specify the combining function
between the intermediate result and accumulator argument of rank K. This
attribute can take the values `add`/`mul`/`minsi`/`minui`/`maxsi`/`maxui`
/`and`/`or`/`xor` for integers, and `add`/`mul`/`minnumf`/`maxnumf`
/`minimumf`/`maximumf` for floats. The default is `add`.

# Example

```mlir
// Simple DOT product (K = 0).
#contraction_accesses = [
 affine_map<(i) -> (i)>,
 affine_map<(i) -> (i)>,
 affine_map<(i) -> ()>
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = [\"reduction\"]
}
%3 = vector.contract #contraction_trait %0, %1, %2
  : vector<10xf32>, vector<10xf32> into f32

// 2D vector contraction with one contracting dimension (matmul, K = 2).
#contraction_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = [\"parallel\", \"parallel\", \"reduction\"]
}

%3 = vector.contract #contraction_trait %0, %1, %2
  : vector<4x3xf32>, vector<3x7xf32> into vector<4x7xf32>

// 4D to 3D vector contraction with two contracting dimensions and
// one batch dimension (K = 3).
#contraction_accesses = [
  affine_map<(b0, f0, f1, c0, c1) -> (c0, b0, c1, f0)>,
  affine_map<(b0, f0, f1, c0, c1) -> (b0, c1, c0, f1)>,
  affine_map<(b0, f0, f1, c0, c1) -> (b0, f0, f1)>
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = [\"parallel\", \"parallel\", \"parallel\",
                    \"reduction\", \"reduction\"]
}

%4 = vector.contract #contraction_trait %0, %1, %2
    : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>

// Vector contraction with mixed typed. lhs/rhs have different element
// types than accumulator/result.
%5 = vector.contract #contraction_trait %0, %1, %2
  : vector<10xf16>, vector<10xf16> into f32

// Contract with max (K = 0).
#contraction_accesses = [
 affine_map<(i) -> (i)>,
 affine_map<(i) -> (i)>,
 affine_map<(i) -> ()>
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = [\"reduction\"],
  kind = #vector.kind<maxnumf>
}
%6 = vector.contract #contraction_trait %0, %1, %2
  : vector<10xf32>, vector<10xf32> into f32
```
"""
function contract(lhs::Value, rhs::Value, acc::Value; result_0::IR.Type, indexing_maps, iterator_types, kind=nothing, location=Location())
    results = IR.Type[result_0, ]
    operands = Value[lhs, rhs, acc, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("indexing_maps", indexing_maps), namedattribute("iterator_types", iterator_types), ]
    !isnothing(kind) && push!(attributes, namedattribute("kind", kind))
    
    create_operation(
        "vector.contract", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`create_mask`

Creates and returns a vector mask where elements of the result vector
are set to \'0\' or \'1\', based on whether the element indices are contained
within a hyper-rectangular region specified by the operands. Specifically,
each operand specifies a range [0, operand-value) for a unique dimension in
the vector result. The conjunction of the operand ranges define a
hyper-rectangular region within which elements values are set to 1
(otherwise element values are set to 0). If operand-value is negative, it is
treated as if it were zero, and if it is greater than the corresponding
dimension size, it is treated as if it were equal to the dimension size.

# Example

```mlir
// create a vector mask of size 4x3xi1 where elements in range
// 0 <= row <= 2 and 0 <= col <= 1 are set to 1 (others to 0).
%1 = vector.create_mask %c3, %c2 : vector<4x3xi1>

print %1
              columns
            0    1    2
          |------------
        0 | 1    1    0
  rows  1 | 1    1    0
        2 | 1    1    0
        3 | 0    0    0
```
"""
function create_mask(operands::Vector{Value}; result_0::IR.Type, location=Location())
    results = IR.Type[result_0, ]
    operands = Value[operands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "vector.create_mask", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`expandload`

The expand load reads elements from memory into a 1-D vector as defined
by a base with indices and a 1-D mask vector. When the mask is set, the
next element is read from memory. Otherwise, the corresponding element
is taken from a 1-D pass-through vector. Informally the semantics are:
```
index = i
result[0] := if mask[0] then base[index++] else pass_thru[0]
result[1] := if mask[1] then base[index++] else pass_thru[1]
etc.
```
Note that the index increment is done conditionally.

If a mask bit is set and the corresponding index is out-of-bounds for the
given base, the behavior is undefined. If a mask bit is not set, the value
comes from the pass-through vector regardless of the index, and the index is
allowed to be out-of-bounds.

The expand load can be used directly where applicable, or can be used
during progressively lowering to bring other memory operations closer to
hardware ISA support for an expand. The semantics of the operation closely
correspond to those of the `llvm.masked.expandload`
[intrinsic](https://llvm.org/docs/LangRef.html#llvm-masked-expandload-intrinsics).

Examples:

```mlir
%0 = vector.expandload %base[%i], %mask, %pass_thru
   : memref<?xf32>, vector<8xi1>, vector<8xf32> into vector<8xf32>

%1 = vector.expandload %base[%i, %j], %mask, %pass_thru
   : memref<?x?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
```
"""
function expandload(base::Value, indices::Vector{Value}, mask::Value, pass_thru::Value; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[base, indices..., mask, pass_thru, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "vector.expandload", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`extractelement`

Takes a 0-D or 1-D vector and a optional dynamic index position and
extracts the scalar at that position.

Note that this instruction resembles vector.extract, but is restricted to
0-D and 1-D vectors and relaxed to dynamic indices.
If the vector is 0-D, the position must be std::nullopt.


It is meant to be closer to LLVM\'s version:
https://llvm.org/docs/LangRef.html#extractelement-instruction

# Example

```mlir
%c = arith.constant 15 : i32
%1 = vector.extractelement %0[%c : i32]: vector<16xf32>
%2 = vector.extractelement %z[]: vector<f32>
```
"""
function extractelement(vector::Value, position=nothing::Union{Nothing, Value}; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[vector, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(position) && push!(operands, position)
    !isnothing(result) && push!(results, result)
    
    create_operation(
        "vector.extractelement", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`extract`

Takes an n-D vector and a k-D position and extracts the (n-k)-D vector at
the proper position. Degenerates to an element type if n-k is zero.

# Example

```mlir
%1 = vector.extract %0[3]: vector<8x16xf32> from vector<4x8x16xf32>
%2 = vector.extract %0[2, 1, 3]: f32 from vector<4x8x16xf32>
%3 = vector.extract %1[]: vector<f32> from vector<f32>
%4 = vector.extract %0[%a, %b, %c]: f32 from vector<4x8x16xf32>
%5 = vector.extract %0[2, %b]: vector<16xf32> from vector<4x8x16xf32>
```
"""
function extract(vector::Value, dynamic_position::Vector{Value}; result=nothing::Union{Nothing, IR.Type}, static_position, location=Location())
    results = IR.Type[]
    operands = Value[vector, dynamic_position..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("static_position", static_position), ]
    !isnothing(result) && push!(results, result)
    
    create_operation(
        "vector.extract", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`extract_strided_slice`

Takes an n-D vector, k-D `offsets` integer array attribute, a k-sized
`sizes` integer array attribute, a k-sized `strides` integer array
attribute and extracts the n-D subvector at the proper offset.

At the moment strides must contain only 1s.

Returns an n-D vector where the first k-D dimensions match the `sizes`
attribute. The returned subvector contains the elements starting at offset
`offsets` and ending at `offsets + sizes`.

# Example

```mlir
%1 = vector.extract_strided_slice %0
    {offsets = [0, 2], sizes = [2, 4], strides = [1, 1]}:
  vector<4x8x16xf32> to vector<2x4x16xf32>

// TODO: Evolve to a range form syntax similar to:
%1 = vector.extract_strided_slice %0[0:2:1][2:4:1]
  vector<4x8x16xf32> to vector<2x4x16xf32>
```
"""
function extract_strided_slice(vector::Value; result_0::IR.Type, offsets, sizes, strides, location=Location())
    results = IR.Type[result_0, ]
    operands = Value[vector, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("offsets", offsets), namedattribute("sizes", sizes), namedattribute("strides", strides), ]
    
    create_operation(
        "vector.extract_strided_slice", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`fma`

Multiply-add expressions operate on n-D vectors and compute a fused
pointwise multiply-and-accumulate: `\$result = `\$lhs * \$rhs + \$acc`.
All operands and result have the same vector type. The semantics
of the operation correspond to those of the `llvm.fma`
[intrinsic](https://llvm.org/docs/LangRef.html#int-fma). In the
particular case of lowering to LLVM, this is guaranteed to lower
to the `llvm.fma.*` intrinsic.

# Example

```mlir
%3 = vector.fma %0, %1, %2: vector<8x16xf32>
```
"""
function fma(lhs::Value, rhs::Value, acc::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs, acc, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    create_operation(
        "vector.fma", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`flat_transpose`

This is the counterpart of llvm.matrix.transpose in MLIR. It serves
the purposes of more progressive lowering and localized type conversion.
Higher levels typically lower matrix tranpositions into \'vector.transpose\'
operations. Subsequent rewriting rule progressively lower these operations
into \'vector.flat_transpose\' operations to bring the operations closer
to the hardware ISA.

The `vector.flat_transpose` op treats the 1-D input `matrix` as
a 2-D matrix with <rows> rows and <columns> columns, and returns the
transposed matrix in flattened form in \'res\'.

Also see:

http://llvm.org/docs/LangRef.html#llvm-matrix-transpose-intrinsic

# Example

```mlir
%1 = vector.flat_transpose %0 {columns = 4 : i32, rows = 4 : i32}
   : vector<16xf32> -> vector<16xf32>
```
"""
function flat_transpose(matrix::Value; res::IR.Type, rows, columns, location=Location())
    results = IR.Type[res, ]
    operands = Value[matrix, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("rows", rows), namedattribute("columns", columns), ]
    
    create_operation(
        "vector.flat_transpose", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`gather`

The gather operation returns an n-D vector whose elements are either loaded
from memory or ranked tensor, or taken from a pass-through vector, depending
on the values of an n-D mask vector.
If a mask bit is set, the corresponding result element is defined by the base
with indices and the n-D index vector (each index is a 1-D offset on the base).
Otherwise, the corresponding element is taken from the n-D pass-through vector.
Informally the semantics are:
```
result[0] := if mask[0] then base[index[0]] else pass_thru[0]
result[1] := if mask[1] then base[index[1]] else pass_thru[1]
etc.
```

If a mask bit is set and the corresponding index is out-of-bounds for the
given base, the behavior is undefined. If a mask bit is not set, the value
comes from the pass-through vector regardless of the index, and the index is
allowed to be out-of-bounds.

The gather operation can be used directly where applicable, or can be used
during progressively lowering to bring other memory operations closer to
hardware ISA support for a gather.

Examples:

```mlir
%0 = vector.gather %base[%c0][%v], %mask, %pass_thru
   : memref<?xf32>, vector<2x16xi32>, vector<2x16xi1>, vector<2x16xf32> into vector<2x16xf32>

%1 = vector.gather %base[%i, %j][%v], %mask, %pass_thru
   : memref<16x16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
```
"""
function gather(base::Value, indices::Vector{Value}, index_vec::Value, mask::Value, pass_thru::Value; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[base, indices..., index_vec, mask, pass_thru, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "vector.gather", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`insertelement`

Takes a scalar source, a 0-D or 1-D destination vector and a dynamic index
position and inserts the source into the destination at the proper position.

Note that this instruction resembles vector.insert, but is restricted to 0-D
and 1-D vectors and relaxed to dynamic indices.

It is meant to be closer to LLVM\'s version:
https://llvm.org/docs/LangRef.html#insertelement-instruction

# Example

```mlir
%c = arith.constant 15 : i32
%f = arith.constant 0.0f : f32
%1 = vector.insertelement %f, %0[%c : i32]: vector<16xf32>
%2 = vector.insertelement %f, %z[]: vector<f32>
```
"""
function insertelement(source::Value, dest::Value, position=nothing::Union{Nothing, Value}; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[source, dest, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(position) && push!(operands, position)
    !isnothing(result) && push!(results, result)
    
    create_operation(
        "vector.insertelement", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`insert`

Takes an n-D source vector, an (n+k)-D destination vector and a k-D position
and inserts the n-D source into the (n+k)-D destination at the proper
position. Degenerates to a scalar or a 0-d vector source type when n = 0.

# Example

```mlir
%2 = vector.insert %0, %1[3] : vector<8x16xf32> into vector<4x8x16xf32>
%5 = vector.insert %3, %4[2, 1, 3] : f32 into vector<4x8x16xf32>
%8 = vector.insert %6, %7[] : f32 into vector<f32>
%11 = vector.insert %9, %10[%a, %b, %c] : vector<f32> into vector<4x8x16xf32>
%12 = vector.insert %4, %10[2, %b] : vector<16xf32> into vector<4x8x16xf32>
```
"""
function insert(source::Value, dest::Value, dynamic_position::Vector{Value}; result=nothing::Union{Nothing, IR.Type}, static_position, location=Location())
    results = IR.Type[]
    operands = Value[source, dest, dynamic_position..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("static_position", static_position), ]
    !isnothing(result) && push!(results, result)
    
    create_operation(
        "vector.insert", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`insert_strided_slice`

Takes a k-D source vector, an n-D destination vector (n >= k), n-sized
`offsets` integer array attribute, a k-sized `strides` integer array attribute
and inserts the k-D source vector as a strided subvector at the proper offset
into the n-D destination vector.

At the moment strides must contain only 1s.

Returns an n-D vector that is a copy of the n-D destination vector in which
the last k-D dimensions contain the k-D source vector elements strided at
the proper location as specified by the offsets.

# Example

```mlir
%2 = vector.insert_strided_slice %0, %1
    {offsets = [0, 0, 2], strides = [1, 1]}:
  vector<2x4xf32> into vector<16x4x8xf32>
```
"""
function insert_strided_slice(source::Value, dest::Value; res=nothing::Union{Nothing, IR.Type}, offsets, strides, location=Location())
    results = IR.Type[]
    operands = Value[source, dest, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("offsets", offsets), namedattribute("strides", strides), ]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "vector.insert_strided_slice", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`interleave`

The interleave operation constructs a new vector by interleaving the
elements from the trailing (or final) dimension of two input vectors,
returning a new vector where the trailing dimension is twice the size.

Note that for the n-D case this differs from the interleaving possible with
`vector.shuffle`, which would only operate on the leading dimension.

Another key difference is this operation supports scalable vectors, though
currently a general LLVM lowering is limited to the case where only the
trailing dimension is scalable.

# Example
```mlir
%0 = vector.interleave %a, %b
           : vector<[4]xi32>     ; yields vector<[8]xi32>
%1 = vector.interleave %c, %d
           : vector<8xi8>        ; yields vector<16xi8>
%2 = vector.interleave %e, %f
           : vector<f16>         ; yields vector<2xf16>
%3 = vector.interleave %g, %h
           : vector<2x4x[2]xf64> ; yields vector<2x4x[4]xf64>
%4 = vector.interleave %i, %j
           : vector<6x3xf32>     ; yields vector<6x6xf32>
```
"""
function interleave(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    create_operation(
        "vector.interleave", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`load`

The \'vector.load\' operation reads an n-D slice of memory into an n-D
vector. It takes a \'base\' memref, an index for each memref dimension and a
result vector type as arguments. It returns a value of the result vector
type. The \'base\' memref and indices determine the start memory address from
which to read. Each index provides an offset for each memref dimension
based on the element type of the memref. The shape of the result vector
type determines the shape of the slice read from the start memory address.
The elements along each dimension of the slice are strided by the memref
strides. Only unit strides are allowed along the most minor memref
dimension. These constraints guarantee that elements read along the first
dimension of the slice are contiguous in memory.

The memref element type can be a scalar or a vector type. If the memref
element type is a scalar, it should match the element type of the result
vector. If the memref element type is vector, it should match the result
vector type.

# Example 0-D vector load on a scalar memref.
```mlir
%result = vector.load %base[%i, %j] : memref<100x100xf32>, vector<f32>
```

# Example 1-D vector load on a scalar memref.
```mlir
%result = vector.load %base[%i, %j] : memref<100x100xf32>, vector<8xf32>
```

# Example 1-D vector load on a vector memref.
```mlir
%result = vector.load %memref[%i, %j] : memref<200x100xvector<8xf32>>, vector<8xf32>
```

# Example  2-D vector load on a scalar memref.
```mlir
%result = vector.load %memref[%i, %j] : memref<200x100xf32>, vector<4x8xf32>
```

# Example  2-D vector load on a vector memref.
```mlir
%result = vector.load %memref[%i, %j] : memref<200x100xvector<4x8xf32>>, vector<4x8xf32>
```

Representation-wise, the \'vector.load\' operation permits out-of-bounds
reads. Support and implementation of out-of-bounds vector loads is
target-specific. No assumptions should be made on the value of elements
loaded out of bounds. Not all targets may support out-of-bounds vector
loads.

# Example  Potential out-of-bound vector load.
```mlir
%result = vector.load %memref[%index] : memref<?xf32>, vector<8xf32>
```

# Example  Explicit out-of-bound vector load.
```mlir
%result = vector.load %memref[%c0] : memref<7xf32>, vector<8xf32>
```
"""
function load(base::Value, indices::Vector{Value}; result::IR.Type, nontemporal=nothing, location=Location())
    results = IR.Type[result, ]
    operands = Value[base, indices..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(nontemporal) && push!(attributes, namedattribute("nontemporal", nontemporal))
    
    create_operation(
        "vector.load", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`mask`

The `vector.mask` is a `MaskingOpInterface` operation that predicates the
execution of another operation. It takes an `i1` vector mask and an
optional passthru vector as arguments.

A implicitly `vector.yield`-terminated region encloses the operation to be
masked. Values used within the region are captured from above. Only one
*maskable* operation can be masked with a `vector.mask` operation at a time.
An operation is *maskable* if it implements the `MaskableOpInterface`. The
terminator yields all results of the maskable operation to the result of
this operation.

The vector mask argument holds a bit for each vector lane and determines
which vector lanes should execute the maskable operation and which ones
should not. The `vector.mask` operation returns the value produced by the
masked execution of the nested operation, if any. The masked-off lanes in
the result vector are taken from the corresponding lanes of the pass-thru
argument, if provided, or left unmodified, otherwise.

The `vector.mask` operation does not prescribe how a maskable operation
should be masked or how a masked operation should be lowered. Masking
constraints and some semantic details are provided by each maskable
operation through the `MaskableOpInterface`. Lowering of masked operations
is implementation defined. For instance, scalarizing the masked operation
or executing the operation for the masked-off lanes are valid lowerings as
long as the execution of masked-off lanes does not change the observable
behavior of the program.

Examples:

```
  %0 = vector.mask %mask { vector.reduction <add>, %a : vector<8xi32> into i32 } : vector<8xi1> -> i32
```

```
  %0 = vector.mask %mask, %passthru { arith.divsi %a, %b : vector<8xi32> } : vector<8xi1> -> vector<8xi32>
```

```
  vector.mask %mask { vector.transfer_write %val, %t0[%idx] : vector<16xf32>, memref<?xf32> } : vector<16xi1>
```

```
  vector.mask %mask { vector.transfer_write %val, %t0[%idx] : vector<16xf32>, tensor<?xf32> } : vector<16xi1> -> tensor<?xf32>
```
"""
function mask(mask::Value, passthru=nothing::Union{Nothing, Value}; results::Vector{IR.Type}, maskRegion::Region, location=Location())
    results = IR.Type[results..., ]
    operands = Value[mask, ]
    owned_regions = Region[maskRegion, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(passthru) && push!(operands, passthru)
    
    create_operation(
        "vector.mask", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`maskedload`

The masked load reads elements from memory into a 1-D vector as defined
by a base with indices and a 1-D mask vector. When the mask is set, the
element is read from memory. Otherwise, the corresponding element is taken
from a 1-D pass-through vector. Informally the semantics are:
```
result[0] := if mask[0] then base[i + 0] else pass_thru[0]
result[1] := if mask[1] then base[i + 1] else pass_thru[1]
etc.
```

If a mask bit is set and the corresponding index is out-of-bounds for the
given base, the behavior is undefined. If a mask bit is not set, the value
comes from the pass-through vector regardless of the index, and the index is
allowed to be out-of-bounds.

The masked load can be used directly where applicable, or can be used
during progressively lowering to bring other memory operations closer to
hardware ISA support for a masked load. The semantics of the operation
closely correspond to those of the `llvm.masked.load`
[intrinsic](https://llvm.org/docs/LangRef.html#llvm-masked-load-intrinsics).

Examples:

```mlir
%0 = vector.maskedload %base[%i], %mask, %pass_thru
   : memref<?xf32>, vector<8xi1>, vector<8xf32> into vector<8xf32>

%1 = vector.maskedload %base[%i, %j], %mask, %pass_thru
   : memref<?x?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
```
"""
function maskedload(base::Value, indices::Vector{Value}, mask::Value, pass_thru::Value; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[base, indices..., mask, pass_thru, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "vector.maskedload", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`maskedstore`

The masked store operation writes elements from a 1-D vector into memory
as defined by a base with indices and a 1-D mask vector. When the mask is
set, the corresponding element from the vector is written to memory. Otherwise,
no action is taken for the element. Informally the semantics are:
```
if (mask[0]) base[i+0] = value[0]
if (mask[1]) base[i+1] = value[1]
etc.
```

If a mask bit is set and the corresponding index is out-of-bounds for the
given base, the behavior is undefined. If a mask bit is not set, no value
is stored regardless of the index, and the index is allowed to be
out-of-bounds.

The masked store can be used directly where applicable, or can be used
during progressively lowering to bring other memory operations closer to
hardware ISA support for a masked store. The semantics of the operation
closely correspond to those of the `llvm.masked.store`
[intrinsic](https://llvm.org/docs/LangRef.html#llvm-masked-store-intrinsics).

Examples:

```mlir
vector.maskedstore %base[%i], %mask, %value
  : memref<?xf32>, vector<8xi1>, vector<8xf32>

vector.maskedstore %base[%i, %j], %mask, %value
  : memref<?x?xf32>, vector<16xi1>, vector<16xf32>
```
"""
function maskedstore(base::Value, indices::Vector{Value}, mask::Value, valueToStore::Value; location=Location())
    results = IR.Type[]
    operands = Value[base, indices..., mask, valueToStore, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "vector.maskedstore", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`matrix_multiply`

This is the counterpart of llvm.matrix.multiply in MLIR. It serves the
purposes of more progressive lowering and localized type conversion.
Higher levels typically lower matrix multiplications into \'vector.contract\'
operations. Subsequent rewriting rule progressively lower these operations
into \'vector.matrix_multiply\' operations to bring the operations closer
to the hardware ISA.

The ‘vector.matrix_multiply’ op treats `lhs` as matrix with <lhs_rows> rows
and <lhs_columns> columns, `rhs` as matrix with <lhs_columns> rows and
<rhs_columns> and multiplies them. The result matrix is returned embedded in
the result vector.

Also see:

http://llvm.org/docs/LangRef.html#llvm-matrix-multiply-intrinsic

# Example

```mlir
%C = vector.matrix_multiply %A, %B
  { lhs_rows = 4: i32, lhs_columns = 16: i32 , rhs_columns = 3: i32 } :
  (vector<64xf64>, vector<48xf64>) -> vector<12xf64>
```
"""
function matrix_multiply(lhs::Value, rhs::Value; res::IR.Type, lhs_rows, lhs_columns, rhs_columns, location=Location())
    results = IR.Type[res, ]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("lhs_rows", lhs_rows), namedattribute("lhs_columns", lhs_columns), namedattribute("rhs_columns", rhs_columns), ]
    
    create_operation(
        "vector.matrix_multiply", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`multi_reduction`

Reduces an n-D vector into an (n-k)-D vector (or a scalar when k == n)
using the given operation: `add`/`mul`/`minsi`/`minui`/`maxsi`/`maxui`
/`and`/`or`/`xor` for integers, and `add`/`mul`/`minnumf`/`maxnumf`/`minimumf`
/`maximumf` for floats.
Takes an initial accumulator operand.

# Example

```mlir
%1 = vector.multi_reduction <add>, %0, %acc0 [1, 3] :
  vector<4x8x16x32xf32> into vector<4x16xf32>
%2 = vector.multi_reduction <add>, %1, %acc1 [0, 1] :
  vector<4x16xf32> into f32
```
"""
function multi_reduction(source::Value, acc::Value; dest=nothing::Union{Nothing, IR.Type}, kind, reduction_dims, location=Location())
    results = IR.Type[]
    operands = Value[source, acc, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("kind", kind), namedattribute("reduction_dims", reduction_dims), ]
    !isnothing(dest) && push!(results, dest)
    
    create_operation(
        "vector.multi_reduction", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`outerproduct`

Takes 2 1-D vectors and returns the 2-D vector containing the outer-product,
as illustrated below:
```
 outer |   [c, d]
 ------+------------
   [a, | [ [a*c, a*d],
    b] |   [b*c, b*d] ]
```
This operation also accepts a 1-D vector lhs and a scalar rhs. In this
case a simple AXPY operation is performed, which returns a 1-D vector.
```
    [a, b] * c = [a*c, b*c]
```

An optional extra vector argument with the same shape as the output
vector may be specified in which case the operation returns the sum of
the outer-product and the extra vector. In this multiply-accumulate
scenario for floating-point arguments, the rounding mode is enforced
by guaranteeing that a fused-multiply add operation is emitted. When
lowered to the LLVMIR dialect, this form emits `llvm.intr.fma`, which
is guaranteed to lower to actual `fma` instructions on x86.

An optional kind attribute may be specified to be: `add`/`mul`/`minsi`
/`minui`/`maxsi`/`maxui`/`and`/`or`/`xor` for integers, and `add`/`mul`
/`minnumf`/`maxnumf`/`minimumf`/`maximumf` for floats. The default is
`add`.

# Example

```
%2 = vector.outerproduct %0, %1: vector<4xf32>, vector<8xf32>
return %2: vector<4x8xf32>

%3 = vector.outerproduct %0, %1, %2:
  vector<4xf32>, vector<8xf32>, vector<4x8xf32>
return %3: vector<4x8xf32>

%4 = vector.outerproduct %0, %1, %2 {kind = #vector.kind<maxnumf>}:
  vector<4xf32>, vector<8xf32>, vector<4x8xf32>
return %3: vector<4x8xf32>

%6 = vector.outerproduct %4, %5: vector<10xf32>, f32
return %6: vector<10xf32>

```
"""
function outerproduct(lhs::Value, rhs::Value, acc=nothing::Union{Nothing, Value}; result_0::IR.Type, kind=nothing, location=Location())
    results = IR.Type[result_0, ]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(acc) && push!(operands, acc)
    !isnothing(kind) && push!(attributes, namedattribute("kind", kind))
    
    create_operation(
        "vector.outerproduct", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`print`

Prints the source vector (or scalar) to stdout in a human-readable format
(for testing and debugging). No return value.

# Example

```mlir
%v = arith.constant dense<0.0> : vector<4xf32>
vector.print %v : vector<4xf32>
```

When lowered to LLVM, the vector print is decomposed into elementary
printing method calls that at runtime will yield:

```
( 0.0, 0.0, 0.0, 0.0 )
```

This is printed to stdout via a small runtime support library, which only
needs to provide a few printing methods (single value for all data
types, opening/closing bracket, comma, newline).

By default `vector.print` adds a newline after the vector, but this can be
controlled by the `punctuation` attribute. For example, to print a comma
after instead do:

```mlir
vector.print %v : vector<4xf32> punctuation <comma>
```

Note that it is possible to use the punctuation attribute alone. The
following will print a single newline:

```mlir
vector.print punctuation <newline>
```

Additionally, to aid with debugging and testing `vector.print` can also
print constant strings:

```mlir
vector.print str \"Hello, World!\"
```
"""
function print(source=nothing::Union{Nothing, Value}; punctuation=nothing, stringLiteral=nothing, location=Location())
    results = IR.Type[]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(source) && push!(operands, source)
    !isnothing(punctuation) && push!(attributes, namedattribute("punctuation", punctuation))
    !isnothing(stringLiteral) && push!(attributes, namedattribute("stringLiteral", stringLiteral))
    
    create_operation(
        "vector.print", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`reduction`

Reduces an 1-D vector \"horizontally\" into a scalar using the given
operation: `add`/`mul`/`minsi`/`minui`/`maxsi`/`maxui`/`and`/`or`/`xor` for
integers, and `add`/`mul`/`minnumf`/`maxnumf`/`minimumf`/`maximumf` for
floats. Reductions also allow an optional fused accumulator.

Note that these operations are restricted to 1-D vectors to remain
close to the corresponding LLVM intrinsics:

http://llvm.org/docs/LangRef.html#vector-reduction-intrinsics

# Example

```mlir
%1 = vector.reduction <add>, %0 : vector<16xf32> into f32

%3 = vector.reduction <xor>, %2 : vector<4xi32> into i32

%4 = vector.reduction <mul>, %0, %1 : vector<16xf32> into f32
```
"""
function reduction(vector::Value, acc=nothing::Union{Nothing, Value}; dest::IR.Type, kind, fastmath=nothing, location=Location())
    results = IR.Type[dest, ]
    operands = Value[vector, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("kind", kind), ]
    !isnothing(acc) && push!(operands, acc)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "vector.reduction", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`reshape`

Reshapes its vector operand from \'input_shape\' to \'output_shape\' maintaining
fixed vector dimension \'fixed_vector_sizes\' on the innermost vector
dimensions.

The parameters \'input_shape\' and \'output_shape\' represent valid data shapes
across fixed vector shapes. For example, if a vector has a valid data
shape [6] with fixed vector size [8], then the valid data elements are
assumed to be stored at the beginning of the vector with the remaining
vector elements undefined.

In the examples below, valid data elements are represented by an alphabetic
character, and undefined data elements are represented by \'-\'.

Example

  vector<1x8xf32> with valid data shape [6], fixed vector sizes [8]

            input: [a, b, c, d, e, f]

       layout map: (d0) -> (d0 floordiv 8, d0 mod 8)

    vector layout: [a, b, c, d, e, f, -, -]

Example

  vector<2x8xf32> with valid data shape [10], fixed vector sizes [8]

            input: [a, b, c, d, e, f, g, h, i, j]

       layout map: (d0) -> (d0 floordiv 8, d0 mod 8)

    vector layout: [[a, b, c, d, e, f, g, h],
                    [i, j, -, -, -, -, -, -]]

Example

  vector<2x2x2x3xf32> with valid data shape [3, 5], fixed vector sizes
  [2, 3]

            input: [[a, b, c, d, e],
                    [f, g, h, i, j],
                    [k, l, m, n, o]]

       layout map: (d0, d1) -> (d0 floordiv 3, d1 floordiv 5,
                                d0 mod 3, d1 mod 5)

    vector layout: [[[[a, b, c],
                      [f, g, h]]
                     [[d, e, -],
                      [i, j, -]]],
                    [[[k, l, m],
                      [-, -, -]]
                     [[n, o, -],
                      [-, -, -]]]]

Example

  %1 = vector.reshape %0, [%c3, %c6], [%c2, %c9], [4]
    : vector<3x2x4xf32> to vector<2x3x4xf32>

         input: [[a, b, c, d, e, f],
                 [g, h, i, j, k, l],
                 [m, n, o, p, q, r]]

    layout map: (d0, d1) -> (d0, d1 floordiv 4, d1 mod 4)


  Input vector:  [[[a, b, c, d],
                   [e, f, -, -]],
                  [[g, h, i, j],
                   [k, l, -, -]],
                  [[m, n, o, p],
                   [q, r, -, -]]]

  Output vector:  [[[a, b, c, d],
                    [e, f, g, h],
                    [i, -, -, -]],
                   [[j, k, l, m],
                    [n, o, p, q],
                    [r, -, -, -]]]
"""
function reshape(vector::Value, input_shape::Vector{Value}, output_shape::Vector{Value}; result::IR.Type, fixed_vector_sizes, location=Location())
    results = IR.Type[result, ]
    operands = Value[vector, input_shape..., output_shape..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("fixed_vector_sizes", fixed_vector_sizes), ]
    push!(attributes, operandsegmentsizes([1, length(input_shape), length(output_shape), ]))
    
    create_operation(
        "vector.reshape", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`scalable_extract`

Takes rank-1 source vector and a position `pos` within the source
vector, and extracts a subvector starting from that position.

The extraction position must be a multiple of the minimum size of the result
vector. For the operation to be well defined, the destination vector must
fit within the source vector from the specified position. Since the source
vector is scalable and its runtime length is unknown, the validity of the
operation can\'t be verified nor guaranteed at compile time.

# Example

```mlir
%1 = vector.scalable.extract %0[8] : vector<4xf32> from vector<[8]xf32>
%3 = vector.scalable.extract %2[0] : vector<[4]xf32> from vector<[8]xf32>
```

Invalid example:
```mlir
%1 = vector.scalable.extract %0[5] : vector<4xf32> from vector<[16]xf32>
```
"""
function scalable_extract(source::Value; res::IR.Type, pos, location=Location())
    results = IR.Type[res, ]
    operands = Value[source, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("pos", pos), ]
    
    create_operation(
        "vector.scalable.extract", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`scalable_insert`

This operations takes a rank-1 fixed-length or scalable subvector and
inserts it within the destination scalable vector starting from the
position specificed by `pos`. If the source vector is scalable, the
insertion position will be scaled by the runtime scaling factor of the
source subvector.

The insertion position must be a multiple of the minimum size of the source
vector. For the operation to be well defined, the source vector must fit in
the destination vector from the specified position. Since the destination
vector is scalable and its runtime length is unknown, the validity of the
operation can\'t be verified nor guaranteed at compile time.

# Example

```mlir
%2 = vector.scalable.insert %0, %1[8] : vector<4xf32> into vector<[16]xf32>
%5 = vector.scalable.insert %3, %4[0] : vector<8xf32> into vector<[4]xf32>
%8 = vector.scalable.insert %6, %7[0] : vector<[4]xf32> into vector<[8]xf32>
```

Invalid example:
```mlir
%2 = vector.scalable.insert %0, %1[5] : vector<4xf32> into vector<[16]xf32>
```
"""
function scalable_insert(source::Value, dest::Value; res=nothing::Union{Nothing, IR.Type}, pos, location=Location())
    results = IR.Type[]
    operands = Value[source, dest, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("pos", pos), ]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "vector.scalable.insert", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`scan`

Performs an inclusive/exclusive scan on an n-D vector along a single
dimension returning an n-D result vector using the given
operation (`add`/`mul`/`minsi`/`minui`/`maxsi`/`maxui`/`and`/`or`/`xor` for
integers, and `add`/`mul`/`minnumf`/`maxnumf`/`minimumf`/`maximumf` for
floats), and a specified value for the initial value. The operator returns
the result of scan as well as the result of the last reduction in the scan.

# Example

```mlir
%1:2 = vector.scan <add>, %0, %acc {inclusive = false, reduction_dim = 1 : i64} :
  vector<4x8x16x32xf32>, vector<4x16x32xf32>
```
"""
function scan(source::Value, initial_value::Value; dest=nothing::Union{Nothing, IR.Type}, accumulated_value=nothing::Union{Nothing, IR.Type}, kind, reduction_dim, inclusive, location=Location())
    results = IR.Type[]
    operands = Value[source, initial_value, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("kind", kind), namedattribute("reduction_dim", reduction_dim), namedattribute("inclusive", inclusive), ]
    !isnothing(dest) && push!(results, dest)
    !isnothing(accumulated_value) && push!(results, accumulated_value)
    
    create_operation(
        "vector.scan", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`scatter`

The scatter operation stores elements from a 1-D vector into memory as
defined by a base with indices and an additional 1-D index vector, but
only if the corresponding bit in a 1-D mask vector is set. Otherwise, no
action is taken for that element. Informally the semantics are:
```
if (mask[0]) base[index[0]] = value[0]
if (mask[1]) base[index[1]] = value[1]
etc.
```

If a mask bit is set and the corresponding index is out-of-bounds for the
given base, the behavior is undefined. If a mask bit is not set, no value
is stored regardless of the index, and the index is allowed to be
out-of-bounds.

If the index vector contains two or more duplicate indices, the behavior is
undefined. Underlying implementation may enforce strict sequential
semantics.
TODO: always enforce strict sequential semantics?

The scatter operation can be used directly where applicable, or can be used
during progressively lowering to bring other memory operations closer to
hardware ISA support for a scatter. The semantics of the operation closely
correspond to those of the `llvm.masked.scatter`
[intrinsic](https://llvm.org/docs/LangRef.html#llvm-masked-scatter-intrinsics).

Examples:

```mlir
vector.scatter %base[%c0][%v], %mask, %value
    : memref<?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>

vector.scatter %base[%i, %j][%v], %mask, %value
    : memref<16x16xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>
```
"""
function scatter(base::Value, indices::Vector{Value}, index_vec::Value, mask::Value, valueToStore::Value; location=Location())
    results = IR.Type[]
    operands = Value[base, indices..., index_vec, mask, valueToStore, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "vector.scatter", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`shape_cast`

The shape_cast operation casts between an n-D source vector shape and
a k-D result vector shape (the element type remains the same).

If reducing rank (n > k), result dimension sizes must be a product
of contiguous source dimension sizes.
If expanding rank (n < k), source dimensions must factor into a
contiguous sequence of destination dimension sizes.
Each source dim is expanded (or contiguous sequence of source dims combined)
in source dimension list order (i.e. 0 <= i < n), to produce a contiguous
sequence of result dims (or a single result dim), in result dimension list
order (i.e. 0 <= j < k). The product of all source dimension sizes and all
result dimension sizes must match.

It is currently assumed that this operation does not require moving data,
and that it will be folded away before lowering vector operations.

There is an exception to the folding expectation when targeting
llvm.intr.matrix operations. We need a type conversion back and forth from a
2-D MLIR vector to a 1-D flattened LLVM vector.shape_cast lowering to LLVM
is supported in that particular case, for now.

# Example

```mlir
// Example casting to a lower vector rank.
%1 = vector.shape_cast %0 : vector<5x1x4x3xf32> to vector<20x3xf32>

// Example casting to a higher vector rank.
%3 = vector.shape_cast %2 : vector<10x12x8xf32> to vector<5x2x3x4x8xf32>

```
"""
function shape_cast(source::Value; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[source, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "vector.shape_cast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`shuffle`

The shuffle operation constructs a permutation (or duplication) of elements
from two input vectors, returning a vector with the same element type as
the input and a length that is the same as the shuffle mask. The two input
vectors must have the same element type, same rank , and trailing dimension
sizes and shuffles their values in the
leading dimension (which may differ in size) according to the given mask.
The legality rules are:
* the two operands must have the same element type as the result
  - Either, the two operands and the result must have the same
    rank and trailing dimension sizes, viz. given two k-D operands
            v1 : <s_1 x s_2 x .. x s_k x type> and
            v2 : <t_1 x t_2 x .. x t_k x type>
    we have s_i = t_i for all 1 < i <= k
  - Or, the two operands must be 0-D vectors and the result is a 1-D vector.
* the mask length equals the leading dimension size of the result
* numbering the input vector indices left to right across the operands, all
  mask values must be within range, viz. given two k-D operands v1 and v2
  above, all mask values are in the range [0,s_1+t_1)

# Example

```mlir
%0 = vector.shuffle %a, %b[0, 3]
           : vector<2xf32>, vector<2xf32>       ; yields vector<2xf32>
%1 = vector.shuffle %c, %b[0, 1, 2]
           : vector<2x16xf32>, vector<1x16xf32> ; yields vector<3x16xf32>
%2 = vector.shuffle %a, %b[3, 2, 1, 0]
           : vector<2xf32>, vector<2xf32>       ; yields vector<4xf32>
%3 = vector.shuffle %a, %b[0, 1]
           : vector<f32>, vector<f32>           ; yields vector<2xf32>
```
"""
function shuffle(v1::Value, v2::Value; vector=nothing::Union{Nothing, IR.Type}, mask, location=Location())
    results = IR.Type[]
    operands = Value[v1, v2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("mask", mask), ]
    !isnothing(vector) && push!(results, vector)
    
    create_operation(
        "vector.shuffle", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`splat`

Broadcast the operand to all elements of the result vector. The operand is
required to be of integer/index/float type.

# Example

```mlir
%s = arith.constant 10.1 : f32
%t = vector.splat %s : vector<8x16xi32>
```
"""
function splat(input::Value; aggregate::IR.Type, location=Location())
    results = IR.Type[aggregate, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "vector.splat", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`store`

The \'vector.store\' operation writes an n-D vector to an n-D slice of memory.
It takes the vector value to be stored, a \'base\' memref and an index for
each memref dimension. The \'base\' memref and indices determine the start
memory address from which to write. Each index provides an offset for each
memref dimension based on the element type of the memref. The shape of the
vector value to store determines the shape of the slice written from the
start memory address. The elements along each dimension of the slice are
strided by the memref strides. Only unit strides are allowed along the most
minor memref dimension. These constraints guarantee that elements written
along the first dimension of the slice are contiguous in memory.

The memref element type can be a scalar or a vector type. If the memref
element type is a scalar, it should match the element type of the value
to store. If the memref element type is vector, it should match the type
of the value to store.

# Example 0-D vector store on a scalar memref.
```mlir
vector.store %valueToStore, %memref[%i, %j] : memref<200x100xf32>, vector<f32>
```

# Example 1-D vector store on a scalar memref.
```mlir
vector.store %valueToStore, %memref[%i, %j] : memref<200x100xf32>, vector<8xf32>
```

# Example 1-D vector store on a vector memref.
```mlir
vector.store %valueToStore, %memref[%i, %j] : memref<200x100xvector<8xf32>>, vector<8xf32>
```

# Example  2-D vector store on a scalar memref.
```mlir
vector.store %valueToStore, %memref[%i, %j] : memref<200x100xf32>, vector<4x8xf32>
```

# Example  2-D vector store on a vector memref.
```mlir
vector.store %valueToStore, %memref[%i, %j] : memref<200x100xvector<4x8xf32>>, vector<4x8xf32>
```

Representation-wise, the \'vector.store\' operation permits out-of-bounds
writes. Support and implementation of out-of-bounds vector stores are
target-specific. No assumptions should be made on the memory written out of
bounds. Not all targets may support out-of-bounds vector stores.

# Example  Potential out-of-bounds vector store.
```mlir
vector.store %valueToStore, %memref[%index] : memref<?xf32>, vector<8xf32>
```

# Example  Explicit out-of-bounds vector store.
```mlir
vector.store %valueToStore, %memref[%c0] : memref<7xf32>, vector<8xf32>
```
"""
function store(valueToStore::Value, base::Value, indices::Vector{Value}; nontemporal=nothing, location=Location())
    results = IR.Type[]
    operands = Value[valueToStore, base, indices..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(nontemporal) && push!(attributes, namedattribute("nontemporal", nontemporal))
    
    create_operation(
        "vector.store", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`transfer_read`

The `vector.transfer_read` op performs a read from a slice within a
[MemRef](../LangRef.md#memref-type) or a Ranked
[Tensor](../LangRef.md#tensor-type) supplied as its first operand
into a [vector](../LangRef.md#vector-type) of the same base elemental type.

A memref/tensor operand with vector element type, must have its vector
element type match a suffix (shape and element type) of the vector (e.g.
memref<3x2x6x4x3xf32>, vector<1x1x4x3xf32>).

The slice is further defined by a full-rank index within the MemRef/Tensor,
supplied as the operands `[1 .. 1 + rank(memref/tensor))` that defines the
starting point of the transfer (e.g. `%A[%i0, %i1, %i2]`).

The permutation_map [attribute](../LangRef.md#attributes) is an
[affine-map](Affine.md#affine-maps) which specifies the transposition on the
slice to match the vector shape. The permutation map may be implicit and
omitted from parsing and printing if it is the canonical minor identity map
(i.e. if it does not permute or broadcast any dimension).

The size of the slice is specified by the size of the vector, given as the
return type.

An SSA value `padding` of the same elemental type as the MemRef/Tensor is
provided to specify a fallback value in the case of out-of-bounds accesses
and/or masking.

An optional SSA value `mask` may be specified to mask out elements read from
the MemRef/Tensor. The `mask` type is an `i1` vector with a shape that
matches how elements are read from the MemRef/Tensor, *before* any
permutation or broadcasting. Elements whose corresponding mask element is
`0` are masked out and replaced with `padding`.

An optional boolean array attribute `in_bounds` specifies for every vector
dimension if the transfer is guaranteed to be within the source bounds. If
specified, the `in_bounds` array length has to be equal to the vector rank.
If set to \"false\", accesses (including the starting point) may run
out-of-bounds along the respective vector dimension as the index increases.
Broadcast dimensions must always be in-bounds. In absence of the attribute,
accesses along all vector dimensions (except for broadcasts) may run
out-of-bounds. A `vector.transfer_read` can be lowered to a simple load if
all dimensions are specified to be within bounds and no `mask` was
specified. Note that non-vector dimensions *must* always be in-bounds.

This operation is called \'read\' by opposition to \'load\' because the
super-vector granularity is generally not representable with a single
hardware register. A `vector.transfer_read` is thus a mid-level abstraction
that supports super-vectorization with non-effecting padding for full-tile
only operations.

More precisely, let\'s dive deeper into the permutation_map for the following
MLIR:

```mlir
vector.transfer_read %A[%expr1, %expr2, %expr3, %expr4]
  { permutation_map : (d0,d1,d2,d3) -> (d2,0,d0) } :
  memref<?x?x?x?xf32>, vector<3x4x5xf32>
```

This operation always reads a slice starting at `%A[%expr1, %expr2, %expr3,
%expr4]`. The size of the slice can be inferred from the resulting vector
shape and walking back through the permutation map: 3 along d2 and 5 along
d0, so the slice is: `%A[%expr1 : %expr1 + 5, %expr2, %expr3:%expr3 + 3, %expr4]`

That slice needs to be read into a `vector<3x4x5xf32>`. Since the
permutation map is not full rank, there must be a broadcast along vector
dimension `1`.

A notional lowering of vector.transfer_read could generate code resembling:

```mlir
// %expr1, %expr2, %expr3, %expr4 defined before this point
// alloc a temporary buffer for performing the \"gather\" of the slice.
%tmp = memref.alloc() : memref<vector<3x4x5xf32>>
for %i = 0 to 3 {
  affine.for %j = 0 to 4 {
    affine.for %k = 0 to 5 {
      // Note that this load does not involve %j.
      %a = load %A[%expr1 + %k, %expr2, %expr3 + %i, %expr4] : memref<?x?x?x?xf32>
      // Update the temporary gathered slice with the individual element
      %slice = memref.load %tmp : memref<vector<3x4x5xf32>> -> vector<3x4x5xf32>
      %updated = vector.insert %a, %slice[%i, %j, %k] : f32 into vector<3x4x5xf32>
      memref.store %updated, %temp : memref<vector<3x4x5xf32>>
}}}
// At this point we gathered the elements from the original
// memref into the desired vector layout, stored in the `%tmp` allocation.
%vec = memref.load %tmp : memref<vector<3x4x5xf32>> -> vector<3x4x5xf32>
```

On a GPU one could then map `i`, `j`, `k` to blocks and threads. Notice that
the temporary storage footprint could conceptually be only `3 * 5` values but
`3 * 4 * 5` values are actually transferred between `%A` and `%tmp`.

Alternatively, if a notional vector broadcast operation were available, we
could avoid the loop on `%j` and the lowered code would resemble:

```mlir
// %expr1, %expr2, %expr3, %expr4 defined before this point
%tmp = memref.alloc() : memref<vector<3x4x5xf32>>
for %i = 0 to 3 {
  affine.for %k = 0 to 5 {
    %a = load %A[%expr1 + %k, %expr2, %expr3 + %i, %expr4] : memref<?x?x?x?xf32>
    %slice = memref.load %tmp : memref<vector<3x4x5xf32>> -> vector<3x4x5xf32>
    // Here we only store to the first element in dimension one
    %updated = vector.insert %a, %slice[%i, 0, %k] : f32 into vector<3x4x5xf32>
    memref.store %updated, %temp : memref<vector<3x4x5xf32>>
}}
// At this point we gathered the elements from the original
// memref into the desired vector layout, stored in the `%tmp` allocation.
// However we haven\'t replicated them alongside the first dimension, we need
// to broadcast now.
%partialVec = load %tmp : memref<vector<3x4x5xf32>> -> vector<3x4x5xf32>
%vec = broadcast %tmpvec, 1 : vector<3x4x5xf32>
```

where `broadcast` broadcasts from element 0 to all others along the
specified dimension. This time, the number of loaded element is `3 * 5`
values.
An additional `1` broadcast is required. On a GPU this broadcast could be
implemented using a warp-shuffle if loop `j` were mapped to `threadIdx.x`.

Syntax
```
operation ::= ssa-id `=` `vector.transfer_read` ssa-use-list
  `{` attribute-entry `} :` memref-type `,` vector-type
```

# Example

```mlir
// Read the slice `%A[%i0, %i1:%i1+256, %i2:%i2+32]` into vector<32x256xf32>
// and pad with %f0 to handle the boundary case:
%f0 = arith.constant 0.0f : f32
affine.for %i0 = 0 to %0 {
  affine.for %i1 = 0 to %1 step 256 {
    affine.for %i2 = 0 to %2 step 32 {
      %v = vector.transfer_read %A[%i0, %i1, %i2], (%f0)
           {permutation_map: (d0, d1, d2) -> (d2, d1)} :
           memref<?x?x?xf32>, vector<32x256xf32>
}}}

// or equivalently (rewrite with vector.transpose)
%f0 = arith.constant 0.0f : f32
affine.for %i0 = 0 to %0 {
  affine.for %i1 = 0 to %1 step 256 {
    affine.for %i2 = 0 to %2 step 32 {
      %v0 = vector.transfer_read %A[%i0, %i1, %i2], (%f0)
           {permutation_map: (d0, d1, d2) -> (d1, d2)} :
           memref<?x?x?xf32>, vector<256x32xf32>
      %v = vector.transpose %v0, [1, 0] :
          vector<256x32xf32> to vector<32x256f32>
}}}

// Read the slice `%A[%i0, %i1]` (i.e. the element `%A[%i0, %i1]`) into
// vector<128xf32>. The underlying implementation will require a 1-D vector
// broadcast:
affine.for %i0 = 0 to %0 {
  affine.for %i1 = 0 to %1 {
    %3 = vector.transfer_read %A[%i0, %i1]
         {permutation_map: (d0, d1) -> (0)} :
         memref<?x?xf32>, vector<128xf32>
  }
}

// Read from a memref with vector element type.
%4 = vector.transfer_read %arg1[%c3, %c3], %vf0
  {permutation_map = (d0, d1)->(d0, d1)}
    : memref<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>

// Read from a tensor with vector element type.
%4 = vector.transfer_read %arg1[%c3, %c3], %vf0
  {permutation_map = (d0, d1)->(d0, d1)}
    : tensor<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>

// Special encoding for 0-d transfer with 0-d tensor/memref, vector shape
// {1} and permutation_map () -> (0).
%0 = vector.transfer_read %arg0[], %f0 {permutation_map = affine_map<()->(0)>} :
  tensor<f32>, vector<1xf32>
```
"""
function transfer_read(source::Value, indices::Vector{Value}, padding::Value, mask=nothing::Union{Nothing, Value}; vector::IR.Type, permutation_map, in_bounds=nothing, location=Location())
    results = IR.Type[vector, ]
    operands = Value[source, indices..., padding, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("permutation_map", permutation_map), ]
    !isnothing(mask) && push!(operands, mask)
    push!(attributes, operandsegmentsizes([1, length(indices), 1, (mask==nothing) ? 0 : 1]))
    !isnothing(in_bounds) && push!(attributes, namedattribute("in_bounds", in_bounds))
    
    create_operation(
        "vector.transfer_read", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`transfer_write`

The `vector.transfer_write` op performs a write from a
[vector](../LangRef.md#vector-type), supplied as its first operand, into a
slice within a [MemRef](../LangRef.md#memref-type) or a Ranked
[Tensor](../LangRef.md#tensor-type) of the same base elemental type,
supplied as its second operand.

A vector memref/tensor operand must have its vector element type match a
suffix (shape and element type) of the vector (e.g. memref<3x2x6x4x3xf32>,
vector<1x1x4x3xf32>). If the operand is a tensor, the operation returns a
new tensor of the same type.

The slice is further defined by a full-rank index within the MemRef/Tensor,
supplied as the operands `[2 .. 2 + rank(memref/tensor))` that defines the
starting point of the transfer (e.g. `%A[%i0, %i1, %i2, %i3]`).

The permutation_map [attribute](../LangRef.md#attributes) is an
[affine-map](Affine.md#affine-maps) which specifies the transposition on the
slice to match the vector shape. The permutation map may be implicit and
omitted from parsing and printing if it is the canonical minor identity map
(i.e. if it does not permute any dimension). In contrast to `transfer_read`,
write ops cannot have broadcast dimensions.

The size of the slice is specified by the size of the vector.

An optional SSA value `mask` may be specified to mask out elements written
to the MemRef/Tensor. The `mask` type is an `i1` vector with a shape that
matches how elements are written into the MemRef/Tensor, *after* applying
any permutation. Elements whose corresponding mask element is `0` are
masked out.

An optional boolean array attribute `in_bounds` specifies for every vector
dimension if the transfer is guaranteed to be within the source bounds. If
specified, the `in_bounds` array length has to be equal to the vector rank.
If set to \"false\", accesses (including the starting point) may run
out-of-bounds along the respective vector dimension as the index increases.
In absence of the attribute, accesses along all vector dimensions may run
out-of-bounds. A `vector.transfer_write` can be lowered to a simple store if
all dimensions are specified to be within bounds and no `mask` was
specified. Note that non-vector dimensions *must* always be in-bounds.

This operation is called \'write\' by opposition to \'store\' because the
super-vector granularity is generally not representable with a single
hardware register. A `vector.transfer_write` is thus a
mid-level abstraction that supports super-vectorization with non-effecting
padding for full-tile-only code. It is the responsibility of
`vector.transfer_write`\'s implementation to ensure the memory writes are
valid. Different lowerings may be pertinent depending on the hardware
support.

# Example

```mlir
// write vector<16x32x64xf32> into the slice
//   `%A[%i0, %i1:%i1+32, %i2:%i2+64, %i3:%i3+16]`:
for %i0 = 0 to %0 {
  affine.for %i1 = 0 to %1 step 32 {
    affine.for %i2 = 0 to %2 step 64 {
      affine.for %i3 = 0 to %3 step 16 {
        %val = `ssa-value` : vector<16x32x64xf32>
        vector.transfer_write %val, %A[%i0, %i1, %i2, %i3]
          {permutation_map: (d0, d1, d2, d3) -> (d3, d1, d2)} :
          vector<16x32x64xf32>, memref<?x?x?x?xf32>
}}}}

// or equivalently (rewrite with vector.transpose)
for %i0 = 0 to %0 {
  affine.for %i1 = 0 to %1 step 32 {
    affine.for %i2 = 0 to %2 step 64 {
      affine.for %i3 = 0 to %3 step 16 {
        %val = `ssa-value` : vector<16x32x64xf32>
        %valt = vector.transpose %val, [1, 2, 0] :
              vector<16x32x64xf32> -> vector<32x64x16xf32>
        vector.transfer_write %valt, %A[%i0, %i1, %i2, %i3]
          {permutation_map: (d0, d1, d2, d3) -> (d1, d2, d3)} :
          vector<32x64x16xf32>, memref<?x?x?x?xf32>
}}}}

// write to a memref with vector element type.
vector.transfer_write %4, %arg1[%c3, %c3]
  {permutation_map = (d0, d1)->(d0, d1)}
    : vector<1x1x4x3xf32>, memref<?x?xvector<4x3xf32>>

// return a tensor where the vector is inserted into the source tensor.
%5 = vector.transfer_write %4, %arg1[%c3, %c3]
  {permutation_map = (d0, d1)->(d0, d1)}
    : vector<1x1x4x3xf32>, tensor<?x?xvector<4x3xf32>>

// Special encoding for 0-d transfer with 0-d tensor/memref, vector shape
// {1} and permutation_map () -> (0).
%1 = vector.transfer_write %0, %arg0[] {permutation_map = affine_map<()->(0)>} :
  vector<1xf32>, tensor<f32>
```
"""
function transfer_write(vector::Value, source::Value, indices::Vector{Value}, mask=nothing::Union{Nothing, Value}; result=nothing::Union{Nothing, IR.Type}, permutation_map, in_bounds=nothing, location=Location())
    results = IR.Type[]
    operands = Value[vector, source, indices..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("permutation_map", permutation_map), ]
    !isnothing(mask) && push!(operands, mask)
    push!(attributes, operandsegmentsizes([1, 1, length(indices), (mask==nothing) ? 0 : 1]))
    !isnothing(result) && push!(results, result)
    !isnothing(in_bounds) && push!(attributes, namedattribute("in_bounds", in_bounds))
    
    create_operation(
        "vector.transfer_write", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`transpose`

Takes a n-D vector and returns the transposed n-D vector defined by
the permutation of ranks in the n-sized integer array attribute (in case
of 0-D vectors the array attribute must be empty).

In the operation

```mlir
%1 = vector.transpose %0, [i_1, .., i_n]
  : vector<d_1 x .. x d_n x f32>
  to vector<d_trans[0] x .. x d_trans[n-1] x f32>
```

the `permutation` array [i_1, .., i_n] must be a permutation of [0, .., n-1].

# Example

```mlir
%1 = vector.transpose %0, [1, 0] : vector<2x3xf32> to vector<3x2xf32>

 [ [a, b, c],       [ [a, d],
   [d, e, f] ]  ->    [b, e],
                      [c, f] ]
```
"""
function transpose(vector::Value; result::IR.Type, permutation, location=Location())
    results = IR.Type[result, ]
    operands = Value[vector, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("permutation", permutation), ]
    
    create_operation(
        "vector.transpose", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`type_cast`

Performs a conversion from a memref with scalar element to a memref with a
*single* vector element, copying the shape of the memref to the vector. This
is the minimal viable operation that is required to makeke
super-vectorization operational. It can be seen as a special case of the
`view` operation but scoped in the super-vectorization context.

# Example

```mlir
%A  = memref.alloc() : memref<5x4x3xf32>
%VA = vector.type_cast %A : memref<5x4x3xf32> to memref<vector<5x4x3xf32>>
```
"""
function type_cast(memref::Value; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[memref, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "vector.type_cast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`warp_execute_on_lane_0`

`warp_execute_on_lane_0` is an operation used to bridge the gap between
vector programming and SPMD programming model like GPU SIMT. It allows to
trivially convert a region of vector code meant to run on a multiple threads
into a valid SPMD region and then allows incremental transformation to
distribute vector operations on the threads.

Any code present in the region would only be executed on first thread/lane
based on the `laneid` operand. The `laneid` operand is an integer ID between
[0, `warp_size`). The `warp_size` attribute indicates the number of lanes in
a warp.

Operands are vector values distributed on all lanes that may be used by
the single lane execution. The matching region argument is a vector of all
the values of those lanes available to the single active lane. The
distributed dimension is implicit based on the shape of the operand and
argument. the properties of the distribution may be described by extra
attributes (e.g. affine map).

Return values are distributed on all lanes using laneId as index. The
vector is distributed based on the shape ratio between the vector type of
the yield and the result type.
If the shapes are the same this means the value is broadcasted to all lanes.
In the future the distribution can be made more explicit using affine_maps
and will support having multiple Ids.

Therefore the `warp_execute_on_lane_0` operations allow to implicitly copy
between lane0 and the lanes of the warp. When distributing a vector
from lane0 to all the lanes, the data are distributed in a block cyclic way.
For exemple `vector<64xf32>` gets distributed on 32 threads and map to
`vector<2xf32>` where thread 0 contains vector[0] and vector[1].

During lowering values passed as operands and return value need to be
visible to different lanes within the warp. This would usually be done by
going through memory.

The region is *not* isolated from above. For values coming from the parent
region not going through operands only the lane 0 value will be accesible so
it generally only make sense for uniform values.

# Example
```
// Execute in parallel on all threads/lanes.
vector.warp_execute_on_lane_0 (%laneid)[32] {
  // Serial code running only on thread/lane 0.
  ...
}
// Execute in parallel on all threads/lanes.
```

This may be lowered to an scf.if region as below:
```
  // Execute in parallel on all threads/lanes.
  %cnd = arith.cmpi eq, %laneid, %c0 : index
  scf.if %cnd {
    // Serial code running only on thread/lane 0.
    ...
  }
  // Execute in parallel on all threads/lanes.
```

When the region has operands and/or return values:
```
// Execute in parallel on all threads/lanes.
%0 = vector.warp_execute_on_lane_0(%laneid)[32]
args(%v0 : vector<4xi32>) -> (vector<1xf32>) {
^bb0(%arg0 : vector<128xi32>) :
  // Serial code running only on thread/lane 0.
  ...
  vector.yield %1 : vector<32xf32>
}
// Execute in parallel on all threads/lanes.
```

values at the region boundary would go through memory:
```
// Execute in parallel on all threads/lanes.
...
// Store the data from each thread into memory and Synchronization.
%tmp0 = memreg.alloc() : memref<128xf32>
%tmp1 = memreg.alloc() : memref<32xf32>
%cnd = arith.cmpi eq, %laneid, %c0 : index
vector.store %v0, %tmp0[%laneid] : memref<128xf32>, vector<4xf32>
some_synchronization_primitive
scf.if %cnd {
  // Serialized code running only on thread 0.
  // Load the data from all the threads into a register from thread 0. This
  // allow threads 0 to access data from all the threads.
  %arg0 = vector.load %tmp0[%c0] : memref<128xf32>, vector<128xf32>
  ...
  // Store the data from thread 0 into memory.
  vector.store %1, %tmp1[%c0] : memref<32xf32>, vector<32xf32>
}
// Synchronization and load the data in a block cyclic way so that the
// vector is distributed on all threads.
some_synchronization_primitive
%0 = vector.load %tmp1[%laneid] : memref<32xf32>, vector<32xf32>
// Execute in parallel on all threads/lanes.
```
"""
function warp_execute_on_lane_0(laneid::Value, args::Vector{Value}; results::Vector{IR.Type}, warp_size, warpRegion::Region, location=Location())
    results = IR.Type[results..., ]
    operands = Value[laneid, args..., ]
    owned_regions = Region[warpRegion, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("warp_size", warp_size), ]
    
    create_operation(
        "vector.warp_execute_on_lane_0", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`yield`

\"vector.yield\" yields an SSA value from the Vector dialect op region and
terminates the regions. The semantics of how the values are yielded is
defined by the parent operation.
If \"vector.yield\" has any operands, the operands must correspond to the
parent operation\'s results.
If the parent operation defines no value the vector.yield may be omitted
when printing the region.
"""
function yield(operands::Vector{Value}; location=Location())
    results = IR.Type[]
    operands = Value[operands..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "vector.yield", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # vector
