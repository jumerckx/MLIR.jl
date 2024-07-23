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

Base.materialize(::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{2}, Nothing, typeof(*), Tuple{MLIRTensor{i64, 2}, MLIRTensor{i64, 2}}}) = op


relu(x::T) where T = max(T(0), x)

cg(Tuple{f64}) do a
    relu(a) >= a
end

cg(Tuple{MLIRTensor{i64, 2}}) do A
    relu(A)
end

f(a, b) = sin.(relu.(a*b))

a = randn(2, 1)

bc1 = Base.broadcasted(relu, a)
bc2 = Base.broadcasted(x->x^2, bc1)

Base.Broadcast.instantiate(bc2)[2, 1]

Base.materialize(bc2)

#################################################

g(a, b) = a .+ b
bc = Base.broadcasted(+, rand(10, 10), rand(10))
Base.Broadcast.instantiate(bc)[9, 7]
