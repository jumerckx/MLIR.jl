module x86vector

import ...IR: IR, NamedAttribute, get_value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`avx_intr_dp_ps_256`

"""
function avx_intr_dp_ps_256(a, b, c; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value(a), get_value(b), get_value(c), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "x86vector.avx.intr.dp.ps.256", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`avx_intr_dot`

Computes the 4-way dot products of the lower and higher parts of the source
vectors and broadcasts the two results to the lower and higher elements of
the destination vector, respectively. Adding one element of the lower part
to one element of the higher part in the destination vector yields the full
dot product of the two source vectors.

# Example

```mlir
%0 = x86vector.avx.intr.dot %a, %b : vector<8xf32>
%1 = vector.extractelement %0[%i0 : i32]: vector<8xf32>
%2 = vector.extractelement %0[%i4 : i32]: vector<8xf32>
%d = arith.addf %1, %2 : f32
```
"""
function avx_intr_dot(a, b; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value(a), get_value(b), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "x86vector.avx.intr.dot", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`avx512_intr_mask_compress`

"""
function avx512_intr_mask_compress(a, src, k; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value(a), get_value(src), get_value(k), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "x86vector.avx512.intr.mask.compress", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`avx512_mask_compress`

The mask.compress op is an AVX512 specific op that can lower to the
`llvm.mask.compress` instruction. Instead of `src`, a constant vector
vector attribute `constant_src` may be specified. If neither `src` nor
`constant_src` is specified, the remaining elements in the result vector are
set to zero.

#### From the Intel Intrinsics Guide:

Contiguously store the active integer/floating-point elements in `a` (those
with their respective bit set in writemask `k`) to `dst`, and pass through the
remaining elements from `src`.
"""
function avx512_mask_compress(k, a, src=nothing; dst=nothing::Union{Nothing, IR.Type}, constant_src=nothing, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value(k), get_value(a), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (src != nothing) && push!(operands, get_value(src))
    !isnothing(dst) && push!(results, dst)
    !isnothing(constant_src) && push!(attributes, namedattribute("constant_src", constant_src))
    
    create_operation(
        "x86vector.avx512.mask.compress", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`avx512_mask_rndscale`

The mask.rndscale op is an AVX512 specific op that can lower to the proper
LLVMAVX512 operation: `llvm.mask.rndscale.ps.512` or
`llvm.mask.rndscale.pd.512` instruction depending on the type of vectors it
is applied to.

#### From the Intel Intrinsics Guide:

Round packed floating-point elements in `a` to the number of fraction bits
specified by `imm`, and store the results in `dst` using writemask `k`
(elements are copied from src when the corresponding mask bit is not set).
"""
function avx512_mask_rndscale(src, k, a, imm, rounding; dst=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value(src), get_value(k), get_value(a), get_value(imm), get_value(rounding), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(dst) && push!(results, dst)
    
    create_operation(
        "x86vector.avx512.mask.rndscale", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`avx512_intr_mask_rndscale_pd_512`

"""
function avx512_intr_mask_rndscale_pd_512(src, k, a, imm, rounding; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value(src), get_value(k), get_value(a), get_value(imm), get_value(rounding), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "x86vector.avx512.intr.mask.rndscale.pd.512", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`avx512_intr_mask_rndscale_ps_512`

"""
function avx512_intr_mask_rndscale_ps_512(src, k, a, imm, rounding; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value(src), get_value(k), get_value(a), get_value(imm), get_value(rounding), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "x86vector.avx512.intr.mask.rndscale.ps.512", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`avx512_mask_scalef`

The `mask.scalef` op is an AVX512 specific op that can lower to the proper
LLVMAVX512 operation: `llvm.mask.scalef.ps.512` or
`llvm.mask.scalef.pd.512` depending on the type of MLIR vectors it is
applied to.

#### From the Intel Intrinsics Guide:

Scale the packed floating-point elements in `a` using values from `b`, and
store the results in `dst` using writemask `k` (elements are copied from src
when the corresponding mask bit is not set).
"""
function avx512_mask_scalef(src, a, b, k, rounding; dst=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value(src), get_value(a), get_value(b), get_value(k), get_value(rounding), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(dst) && push!(results, dst)
    
    create_operation(
        "x86vector.avx512.mask.scalef", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`avx512_intr_mask_scalef_pd_512`

"""
function avx512_intr_mask_scalef_pd_512(src, a, b, k, rounding; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value(src), get_value(a), get_value(b), get_value(k), get_value(rounding), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "x86vector.avx512.intr.mask.scalef.pd.512", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`avx512_intr_mask_scalef_ps_512`

"""
function avx512_intr_mask_scalef_ps_512(src, a, b, k, rounding; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value(src), get_value(a), get_value(b), get_value(k), get_value(rounding), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "x86vector.avx512.intr.mask.scalef.ps.512", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`avx_intr_rsqrt_ps_256`

"""
function avx_intr_rsqrt_ps_256(a; res=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value(a), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    
    create_operation(
        "x86vector.avx.intr.rsqrt.ps.256", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`avx_rsqrt`

"""
function avx_rsqrt(a; b=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value(a), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(b) && push!(results, b)
    
    create_operation(
        "x86vector.avx.rsqrt", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`avx512_intr_vp2intersect_d_512`

"""
function avx512_intr_vp2intersect_d_512(a, b; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = API.MlirValue[get_value(a), get_value(b), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "x86vector.avx512.intr.vp2intersect.d.512", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`avx512_vp2intersect`

The `vp2intersect` op is an AVX512 specific op that can lower to the proper
LLVMAVX512 operation: `llvm.vp2intersect.d.512` or
`llvm.vp2intersect.q.512` depending on the type of MLIR vectors it is
applied to.

#### From the Intel Intrinsics Guide:

Compute intersection of packed integer vectors `a` and `b`, and store
indication of match in the corresponding bit of two mask registers
specified by `k1` and `k2`. A match in corresponding elements of `a` and
`b` is indicated by a set bit in the corresponding bit of the mask
registers.
"""
function avx512_vp2intersect(a, b; k1=nothing::Union{Nothing, IR.Type}, k2=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value(a), get_value(b), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(k1) && push!(results, k1)
    !isnothing(k2) && push!(results, k2)
    
    create_operation(
        "x86vector.avx512.vp2intersect", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`avx512_intr_vp2intersect_q_512`

"""
function avx512_intr_vp2intersect_q_512(a, b; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = API.MlirValue[get_value(a), get_value(b), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "x86vector.avx512.intr.vp2intersect.q.512", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # x86vector
