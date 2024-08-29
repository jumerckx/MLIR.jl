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
    bc = Base.broadcasted(relu, A)
    ElType = Base.Broadcast.combine_eltypes(bc.f, bc.args)
    similar(bc, ElType)
end

cg(Tuple{MLIRTensor{i64, 2}, MLIRTensor{i64, 2}}) do A, B
    A .+ B
end

cg(Tuple{MLIRStaticTensor{i64, 2, Tuple{1,42}}}) do A
    size(A)
end

Base.Broadcast.broadcasted(+, Base.Broadcast.broadcasted(-, rand(10), rand(1, 9)), rand(10, 9))

struct Eval

Base.code_ircode(Tuple{MLIRStaticTensor{i64, 2, Tuple{1,2}}}, interp=Generate.MLIRInterpreter()) do A
    size(A)
end

########## ForwardDiff ##########

using ForwardDiff

@time cg(Tuple{ForwardDiff.Dual{Nothing, f64, 1}}) do a
    a*a
end
