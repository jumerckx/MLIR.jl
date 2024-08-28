using Revise
includet("Einsum.jl")
import MLIR.Generate: @intrinsic, CodegenContext
import MLIR: Dialects, IR, API

# administrative duties
function registerAllDialects!()
    ctx = IR.context()
    registry = API.mlirDialectRegistryCreate()
    API.mlirRegisterAllDialects(registry)
    API.mlirContextAppendDialectRegistry(ctx, registry)
    API.mlirDialectRegistryDestroy(registry)

    API.mlirContextLoadAllAvailableDialects(ctx)
    return registry
end
ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

cg = CodegenContext()

cg(Tuple{i64}) do a
    @noinline f() = a*a
    f()
end

mod = cg(Tuple{i64, i64, i64}) do y, a,b
    execute_region(i64) do 
        y+prod((a, b))
    end
end


CodegenContext{LinalgBody}()(
    (xs, y)-> execute_region(i64) do 
        y+prod(xs)
    end,
    Tuple{Tuple{i64, i64}, i64}
) |> show

op = cg(Tuple{MLIRTensor{i64, 2}, MLIRTensor{i64, 2}, MLIRTensor{i64, 2}}) do Y, A, B
    Einsum(((:i, :k), (:k, :j))=>(:i, :j))(Y, A, B)
end

import LinearAlgebra

@intrinsic function _empty_matmul_result(A::TA, B::TB) where {F, TA <: MLIRTensor{F, 2}, TB <: MLIRTensor{F, 2}}
    N, K = size(IR.value(A))
    _, M = size(IR.value(B))

    dynamicSizes = []
    if IR.isdynsize(N)
        zero = Dialects.index.constant(value=IR.Attribute(0, IR.Type(index)), result=IR.Type(index))|>IR.result
        push!(dynamicSizes, IR.result(Dialects.tensor.dim(A, zero)))
    end
    if IR.isdynsize(M)
        one = Dialects.index.constant(value=IR.Attribute(1, IR.Type(index)), result=IR.Type(index))|>IR.result
        push!(dynamicSizes, IR.result(Dialects.tensor.dim(B, one)))
    end

    Y = MLIRTensor{F, 2}(Dialects.tensor.empty(dynamicSizes, result=IR.TensorType([N, M], IR.Type(F))) |> IR.result)
end

function LinearAlgebra.:*(A::MLIRTensor, B::MLIRTensor)
    Y = _empty_matmul_result(A, B)
    Einsum(((:i, :k), (:k, :j))=>(:i, :j))(Y, A, B)
end
LinearAlgebra.:*(A::MLIRTensor, B::MLIRTensor, C::MLIRTensor) = (A*B)*C
LinearAlgebra.:*(A::MLIRTensor, B::MLIRTensor, C::MLIRTensor, D::MLIRTensor) = ((A*B)*C)*D

@time op = cg(Tuple{MLIRTensor{i64, 2}, MLIRTensor{i64, 2}}) do A, B
    A*B
end

@intrinsic function _elementwise_op!(f::F, Y::T, X::T) where {E, N, T<:MLIRTensor{E, N}, F}
    iterator_type = parse(IR.Attribute, "#linalg.iterator_type<parallel>")
    indexing_map = IR.Attribute(IR.MinorIdentityAffineMap(N, N))
    cg = CodegenContext{LinalgBody}()
    region = cg(Tuple{E, E}) do x, y
        execute_region(E) do
            f(x)
        end
    end
    op = linalg.generic(
        [X],
        [Y];
        result_tensors=IR.Type[IR.Type(T)],
        indexing_maps=[indexing_map, indexing_map],
        iterator_types=[iterator_type, iterator_type],
        region
    )
    return T(IR.result(op))
end

@inline _elementwise_op(f::F, X::T) where {F, T} = _elementwise_op!(f, similar(X), X)

relu(x::T) where T = max(T(0), x)

cg(Tuple{f64}) do a
    relu(a)
end


@time cg(Tuple{MLIRTensor{i64, 2}}) do A
    B = A*A
    _elementwise_op((x->x+x) ∘ relu, B)
end

Base.code_ircode(Tuple{MLIRTensor{i64, 2}}, interp=Generate.MLIRInterpreter()) do A
    relu.(A)
end

cg(Tuple{MLIRTensor{i64, 2}}) do A
    Base.broadcasted(relu, A)[1, 1]
end

cg(Tuple{MLIRTensor{i64, 2}}) do A
    b = Base.broadcasted(relu, A)
    @inbounds Base.Broadcast._broadcast_getindex(b, CartesianIndex(1, 1))
end

cg(Tuple{MLIRTensor{i64, 2}, MLIRTensor{i64, 2}}) do A, B
    # b = Base.broadcasted(+, A, B)
    # axes(b)
    Base.Broadcast._bcs1(axes(A), axes(B))
    # Base.Broadcast._bcsm(axes(A), axes(B))
end

Base.code_ircode(Tuple{MLIRTensor{i64, 2}, MLIRTensor{i64, 2}}, interp=Generate.MLIRInterpreter()) do A, B
    # b = Base.broadcasted(+, A, B)
    # axes(b)
    # length(axes(A)[1])
    Base.Broadcast._bcs1(axes(A), axes(B))
    # Base.Broadcast._bcsm(axes(A), axes(B))
end

Base.code_ircode(Base._eq, Tuple{Tuple{Base.OneTo{MLIRIndex}, Base.OneTo{MLIRIndex}}, Tuple{Base.OneTo{MLIRIndex}, Base.OneTo{MLIRIndex}}}, interp=Generate.MLIRInterpreter())

cg(Tuple{i1}) do c
    !c
end

using Cthulhu
# execute in REPL:
# descend(Tuple{MLIRTensor{i64, 2}}, interp=Generate.MLIRInterpreter()) do A
#     relu.(A)
# end

cg(Tuple{index, index}) do n, m
    similar(MLIRTensor{i64, 1}, (n, m))
end

to_indices(A::MLIRTensor{i64, 2}, I::Tuple{CartesianIndex{2}})::Tuple{Int64, Int64}

cg(Tuple{MLIRTensor{i64, 2}}) do A
    relu.(A)
end

f(a) = relu.(a)

# @enter f(randn(10))

cg = CodegenContext()
region = cg(Tuple{i64}) do el
    execute_region(i64) do
        relu(el)
    end
end

f(a, b) = relu.(a*b)
fib(n) = n < 2 ? n : fib(n-1) + fib(n-2)

cg(Tuple{i64}) do n
    fib(n)
end

cg(Tuple{i64}) do a
    if a > a
        b = a
    else
        b = a-a
    end
    return b
end

using ForwardDiff

@time cg(Tuple{ForwardDiff.Dual{Nothing, f64, 1}}) do a
    a*a
end

a = ForwardDiff.Dual(23, 1)

a*a

ForwardDiff.Dual(10.).partials

Base.code_ircode(Tuple{i64}, interp=Generate.MLIRInterpreter()) do a
    if a > a
        b = a
    else
        b = a-a
    end
    return b
end

a = randn(2, 1)

bc1 = Base.broadcasted(relu, a)
bc2 = Base.broadcasted(x->x^2, bc1)

Base.Broadcast.instantiate(bc2)[2, 1]

Base.materialize(bc2)

#################################################

g(a, b) = relu.(a .+ b)

a, b = randn(10), randn(10, 2)

@enter g(a, b)

bc = Base.broadcasted(+, rand(10, 10), rand(10))
Base.Broadcast.instantiate(bc)[9, 7]
