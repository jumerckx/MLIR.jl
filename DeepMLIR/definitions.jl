import MLIR.IR
using MLIR.IR: Value, Attribute, result, Operation, Convertible, context, IndexType, ValueTrait
import MLIR.Dialects
using MLIR.API: mlirMemRefTypeGet, mlirStridedLayoutAttrGet, mlirRankedTensorTypeGet, mlirIntegerTypeGet, mlirShapedTypeGetDynamicSize, mlirF64TypeGet, mlirF32TypeGet, mlirF16TypeGet
import MLIR.Generate
import MLIR.Generate: @intrinsic, BoolTrait

### int ###
struct MLIRInteger{N} <: Integer
    value::Value
    MLIRInteger{N}(i::Value) where {N} = new(i)
end
ValueTrait(::Type{<:MLIRInteger}) = Convertible()
IR.Type(::Type{MLIRInteger{N}}) where {N} = IR.Type(mlirIntegerTypeGet(context(), N))

const i1 = MLIRInteger{1}
BoolTrait(::Type{i1}) = Generate.Boollike()
@intrinsic Base.:!(a::i1)::i1 = i1(Dialects.arith.xori(a, i1(true))|>result)

const i8 = MLIRInteger{8}
const i16 = MLIRInteger{16}
const i32 = MLIRInteger{32}
const i64 = MLIRInteger{64}

@intrinsic Base.:+(a::T, b::T) where {T<:MLIRInteger} = T(Dialects.arith.addi(a, b)|>result)
@intrinsic Base.:-(a::T, b::T) where {T<:MLIRInteger} = T(Dialects.arith.subi(a, b)|>result)
@intrinsic Base.:*(a::T, b::T) where {T<:MLIRInteger} = T(Dialects.arith.muli(a, b)|>result)
@intrinsic Base.:/(a::T, b::T) where {T<:MLIRInteger} = T(Dialects.arith.divi(a, b)|>result)

@intrinsic Base.:>(a::T, b::T) where {T<:MLIRInteger} = i1(Dialects.arith.cmpi(a, b, result=IR.Type(i1), predicate=4)|>result)
@intrinsic Base.:>=(a::T, b::T) where {T<:MLIRInteger} = i1(Dialects.arith.cmpi(a, b, result=IR.Type(i1), predicate=5)|>result)
@intrinsic Base.:<(a::T, b::T) where {T<:MLIRInteger} = i1(Dialects.arith.cmpi(a, b, result=IR.Type(i1), predicate=2)|>result)
@intrinsic Base.:<=(a::T, b::T) where {T<:MLIRInteger} = i1(Dialects.arith.cmpi(a, b, result=IR.Type(i1), predicate=3)|>result)

@intrinsic Base.min(a::T, b::T) where {T<:MLIRInteger} = T(Dialects.arith.minsi(a, b)|>result)
@intrinsic Base.max(a::T, b::T) where {T<:MLIRInteger} = T(Dialects.arith.maxsi(a, b)|>result)

@intrinsic Base.:&(a::T, b::T) where {T<:MLIRInteger} = T(Dialects.arith.andi(a, b)|>result)

# promote constant julia integers to int
@intrinsic i64(x::Integer) = i64(Dialects.arith.constant(value=Attribute(Int64(x)), result=IR.Type(i64))|>result)
@intrinsic i32(x::Integer) = i32(Dialects.arith.constant(value=Attribute(Int32(x)), result=IR.Type(i32))|>result)
@intrinsic i16(x::Integer) = i16(Dialects.arith.constant(value=Attribute(Int16(x)), result=IR.Type(i16))|>result)
@intrinsic i8(x::Integer) = i8(Dialects.arith.constant(value=Attribute(Int8(x)), result=IR.Type(i8))|>result)
@intrinsic i1(x::Bool) = i1(Dialects.arith.constant(value=Attribute(Int8(x)), result=IR.Type(i1))|>result)

i64(x::i64) = x
i32(x::i32) = x
i16(x::i16) = x
i8(x::i8) = x
i1(x::i1) = x

Base.promote_rule(::Type{T}, ::Type{I}) where {T<:MLIRInteger, I<:Integer} = T
Base.convert(::Type{T}, x::T) where {T <: MLIRInteger} = x
@intrinsic function Base.convert(::Type{T}, x::Integer)::T where {T<:MLIRInteger}
    op = Dialects.arith.constant(value=Attribute(x), result=IR.Type(T))
    T(result(op))
end

### float ###
abstract type MLIRFloat <: AbstractFloat end
ValueTrait(::Type{<:MLIRFloat}) = Convertible()

struct MLIRF64 <: MLIRFloat
    value::Value
end
struct MLIRF32 <: MLIRFloat
    value::Value
end
struct MLIRF16 <: MLIRFloat
    value::Value
end

const f64 = MLIRF64
const f32 = MLIRF32
const f16 = MLIRF16

IR.Type(::Type{MLIRF64}) = IR.Type(mlirF64TypeGet(context()))
IR.Type(::Type{MLIRF32}) = IR.Type(mlirF32TypeGet(context()))
IR.Type(::Type{MLIRF16}) = IR.Type(mlirF16TypeGet(context()))

@intrinsic (Base.:+(a::T, b::T)::T) where {T<:MLIRFloat} = T(Dialects.arith.addf(a, b)|>result)
@intrinsic (Base.:-(a::T, b::T)::T) where {T<:MLIRFloat} = T(Dialects.arith.subf(a, b)|>result)
@intrinsic (Base.:*(a::T, b::T)::T) where {T<:MLIRFloat} = T(Dialects.arith.mulf(a, b)|>result)
@intrinsic (Base.:/(a::T, b::T)::T) where {T<:MLIRFloat} = T(Dialects.arith.divf(a, b)|>result)

# TODO: 
# @intrinsic Base.:>(a::T, b::T)::i1 where {T<:MLIRFloat} = i1(Dialects.arith.cmpf(a, b, predicate=...))
# @intrinsic Base.:>=(a::T, b::T)::i1 where {T<:MLIRFloat} = i1(Dialects.arith.cmpf(a, b, predicate=...))
# @intrinsic Base.:<(a::T, b::T)::i1 where {T<:MLIRFloat} = i1(Dialects.arith.cmpf(a, b, predicate=...))
# @intrinsic Base.:<=(a::T, b::T)::i1 where {T<:MLIRFloat} = i1(Dialects.arith.cmpf(a, b, predicate=...))

@intrinsic Base.min(a::T, b::T) where {T<:MLIRFloat} = T(Dialects.arith.minf(a, b)|>result)
@intrinsic Base.max(a::T, b::T) where {T<:MLIRFloat} = T(Dialects.arith.maxf(a, b)|>result)

f32(x::f32) = x
@intrinsic f32(x::Real) = f32(Dialects.arith.constant(value=IR.Attribute(Float32(x)), result=IR.Type(f32)) |> result)
Base.convert(::Type{f32}, x::Real) = f32(x)
Base.promote_rule(::Type{f32}, ::Type{<:Real}) = f32

f64(x::f64) = x
@intrinsic f64(x::Real) = f64(Dialects.arith.constant(value=IR.Attribute(Float64(x)), result=IR.Type(f64)) |> result)
Base.convert(::Type{f64}, x::Real) = f64(x)
Base.promote_rule(::Type{f64}, ::Type{<:Real}) = f64

### index  ###
struct MLIRIndex <: Integer
    value::Value
end
const index = MLIRIndex
IR.Type(::Type{MLIRIndex}) = IndexType()
ValueTrait(::Type{<:MLIRIndex}) = Convertible()

@intrinsic Base.:+(a::index, b::index)::index = index(Dialects.index.add(a, b)|>result)
@intrinsic Base.:-(a::index, b::index)::index = index(Dialects.index.sub(a, b)|>result)
@intrinsic Base.:*(a::index, b::index)::index = index(Dialects.index.mul(a, b)|>result)
@intrinsic Base.div(a::index, b::index)::index = index(Dialects.index.divs(a, b)|>result)

# TODO:
@intrinsic Base.:>(a::index, b::index)::i1 = i1(Dialects.index.cmp(a, b, pred=4)|>result)
@intrinsic Base.:>=(a::index, b::index)::i1 = i1(Dialects.index.cmp(a, b, pred=5)|>result)
@intrinsic Base.:<(a::index, b::index)::i1 = i1(Dialects.index.cmp(a, b, pred=2)|>result)
@intrinsic Base.:<=(a::index, b::index)::i1 = i1(Dialects.index.cmp(a, b, pred=3)|>result)
@intrinsic Base.:(==)(a::index, b::index)::i1 = i1(Dialects.index.cmp(a, b, pred=0)|>result)

@intrinsic Base.min(a::index, b::index) = index(Dialects.index.mins(a, b)|>result)
@intrinsic Base.max(a::index, b::index) = index(Dialects.index.maxs(a, b)|>result)

# promote constant julia integers to index
@intrinsic index(x::Integer) = index(Dialects.index.constant(value=Attribute(x, IR.Type(index)), result=IR.Type(index))|>result)
index(x::index) = x
Base.promote_rule(::Type{index}, ::Type{I}) where {I<:Integer} = index
function Base.convert(::Type{index}, x::Integer)::index
    index(x)
end

@intrinsic i64(x::index) = i64(Dialects.index.casts(x, output=IR.Type(i64))|>result)
@intrinsic index(x::i64) = index(Dialects.index.casts(x, output=IR.Type(index))|>result)

Base.to_shape(i::index) = i
Base.to_shape(r::Base.OneTo{index}) = index(last(r))

### abstract type for array-like types ###
abstract type MLIRArrayLike{T, N} <: AbstractArray{T, N} end

ValueTrait(::Type{<:MLIRArrayLike}) = Convertible()
Base.show(io::IO, a::A) where {A<:MLIRArrayLike{T, N}} where {T, N} = print(io, "$A[...]")
Base.show(io::IO, ::MIME{Symbol("text/plain")}, a::A) where {A<:MLIRArrayLike{T, N}} where {T, N} = print(io, "$A[...]")

### memref ###
struct MLIRMemref{T, N, Shape, Memspace, Stride, Offset} <: MLIRArrayLike{T, N}
    value::Value
end
function IR.Type(::Type{MLIRMemref{T, N, Shape, Memspace, Stride, Offset}}) where {T, N, Shape, Memspace, Stride, Offset}
    memspace(a::Attribute) = a
    memspace(::Nothing) = Attribute()
    memspace(i::Integer) = Attribute(i)

    shape(::Nothing) = Int[mlirShapedTypeGetDynamicSize() for _ in 1:N]
    shape(s) = Int[s.parameters...]

    # default to column-major layout
    stride(::Nothing) = Int[1, [mlirShapedTypeGetDynamicSize() for _ in 2:N]...]
    stride(s) = shape(s)

    offset(::Nothing) = mlirShapedTypeGetDynamicSize()
    offset(i::Integer) = i

    IR.Type(mlirMemRefTypeGet(
        IR.Type(T),
        N,
        shape(Shape),
        Attribute(mlirStridedLayoutAttrGet(
            context().context,
            offset(Offset),
            N,
            stride(Stride))),
        memspace(Memspace)
    ))

end
const memref{T, N} = MLIRMemref{T, N, nothing, nothing, nothing, 0}

Base.size(A::MLIRMemref{T, N, Shape}) where {T, N, Shape} = Tuple(Shape.parameters)

@intrinsic function Base.getindex(A::MLIRMemref{T, 1}, i::index)::T where T
    oneoff = Dialects.index.constant(; value=Attribute(1, IndexType())) |> result
    new_index = Dialects.index.sub(i, oneoff) |> result
    T(Dialects.memref.load(A, [new_index]) |> result)
end
function Base.getindex(A::MLIRMemref{T}, i::Int)::T where T
    A[index(i)]
end

@intrinsic function Base.setindex!(A::MLIRMemref{T, 1}, v::T, i::index)::T where T
    oneoff = Dialects.index.constant(; value=Attribute(1, IndexType())) |> result
    new_index = Dialects.index.sub(i, oneoff) |> IR.result
    Dialects.memref.store(v, A, [new_index])
    return v
end
@intrinsic function Base.setindex!(A::MLIRMemref{T, 1}, v, i::Int)::T where {T}
    # this method can only be called with constant i since we assume arguments to `code_mlir` to be MLIR types, not Julia types.
    i = index(Dialects.index.constant(; value=Attribute(i, IndexType())) |> result)
    A[i] = v
end

### tensor ###
struct MLIRTensor{T, N} <: MLIRArrayLike{T, N}
    value::Value
end
IR.Type(::Type{MLIRTensor{T, N}}) where {T, N} = mlirRankedTensorTypeGet(
    N,
    Int[mlirShapedTypeGetDynamicSize() for _ in 1:N],
    IR.Type(T),
    Attribute()) |> IR.Type
const tensor = MLIRTensor

@intrinsic function Base.size(A::MLIRTensor{T, N}) where {T, N}
    sizes = []
    for i in 1:N
        s = Dialects.tensor.dim(A, index(i-1))|>result
        push!(sizes, index(s))
    end
    return Tuple(sizes)::NTuple{N, index}
end

@intrinsic function _create_empty_tensor(dims, element_type)
    MLIRTensor{element_type, length(dims)}(Dialects.tensor.empty(
        dims;
        result=IR.TensorType(fill(IR.dynsize(), length(dims)), IR.Type(element_type))
    ) |> result)
end

# inline these definitions because the type argument can't be converted to an argument in MLIR code.
@inline Base.similar(a::MLIRTensor{T}) where {T} = similar(a, T)
@inline Base.similar(a::MLIRTensor, ::Type{T}) where {T} = similar(a, T, Base.to_shape(axes(a)))
@inline Base.similar(::MLIRTensor{T}, ::Type{T}, dims::NTuple{N, index}) where {T, N} = MLIRTensor{T}(undef, dims)

@inline Base.similar(::Type{T}, dims::NTuple{N, index}) where {N, T<:MLIRTensor} = T(undef, dims)

function MLIRTensor{T, N}(::UndefInitializer, dims::NTuple{N, index}) where {T, N}
    _create_empty_tensor(dims, T)
end

# type and dimensionality specified
MLIRTensor{T, N}(::UndefInitializer, dims::Vararg{index, N}) where {T, N} = MLIRTensor{T, N}(undef, convert(Tuple{Vararg{index}}, dims))

# only type specified
MLIRTensor{T}(::UndefInitializer, dims::NTuple{N,index}) where {T, N} = MLIRTensor{T, N}(undef, convert(Tuple{Vararg{index}}, dims))
MLIRTensor{T}(::UndefInitializer, dims::Vararg{index,N}) where {T, N} = MLIRTensor{T, N}(undef, convert(Tuple{Vararg{index}}, dims))



struct MLIRArrayStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end
MLIRArrayStyle(::Val{N}) where {N} = MLIRArrayStyle{N}()

# identify the broadcast style of a MLIRTensor
Base.Broadcast.BroadcastStyle(::Type{<:MLIRTensor{T,N}}) where {T,N} = MLIRArrayStyle{N}()

# don't check for broadcast compatibility because we can't throw errors either way.
Base.Broadcast._bcs1(a::Base.OneTo{index}, b::Base.OneTo{index}) = i1(Base.Broadcast._bcsm(b, a)) ? b : a
function Base._eq(t1::NTuple{N, T}, t2::NTuple{N, T}) where {N, T<:Base.OneTo{index}}
    eq = t1[1] == t2[1]
    if !eq
        return eq
    else
        return i1(Base._eq(Base.tail(t1), Base.tail(t2)))
    end
end

# # when we are dealing with different buffer styles, we cannot know
# # which one is better, so use shared memory
# BroadcastStyle(::MtlArrayStyle{N, S1},
#                ::MtlArrayStyle{N, S2}) where {N,S1,S2} =
#     MtlArrayStyle{N, SharedStorage}()

# allocation of output arrays
Base.similar(bc::Base.Broadcast.Broadcasted{MLIRArrayStyle{N}}, ::Type{T}, dims) where {T,N} =
    similar(MLIRTensor{T,length(dims)}, dims)