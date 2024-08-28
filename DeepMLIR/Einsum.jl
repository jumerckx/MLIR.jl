includet("definitions.jl")

import MLIR: IR
import MLIR.Generate: @intrinsic, CodegenContext, generate_return, generate_function, aggregate_funcs
import MLIR.Dialects: scf, linalg


struct Einsum{T}
    desc::Pair{T}
    @intrinsic function Einsum(desc::Pair{T}) where {T}
        return new{T}(desc)
    end
end

function maps(e::Einsum)
    return maps(e.desc.second, e.desc.first)
end

function maps(output_indices, inputs_indices)
    parallel, reduction =
        parse.(
            Ref(IR.Attribute),
            ("#linalg.iterator_type<parallel>", "#linalg.iterator_type<reduction>"),
        )

    # get all index symbols used in the inputs, the output can't contain any different symbols.
    symbols = Dict()
    iterator_types = IR.Attribute[]
    for arg in inputs_indices
        map(arg) do i
            get!(symbols, i) do
                # indices that don't occur in output have to be reduced
                push!(iterator_types, i ∉ output_indices ? reduction : parallel)

                return API.mlirAffineDimExprGet(IR.context(), length(symbols))
            end
        end
    end

    # function to create affinemap
    function get_map(indices)
        exprs = map(indices) do i
            symbols[i]
        end
        return IR.AffineMap(API.mlirAffineMapGet(
            IR.context(), length(symbols), 0, length(indices), collect(exprs)
        ))
    end

    indexing_maps = IR.AffineMap[get_map.(inputs_indices)..., get_map(output_indices)]

    iterator_types = IR.Attribute(iterator_types)
    indexing_maps = IR.Attribute(IR.Attribute.(API.mlirAffineMapAttrGet.(indexing_maps)))
    return indexing_maps, iterator_types
end

abstract type ExecuteRegion end
generate_return(cg::CodegenContext{ExecuteRegion}, values; location) = scf.yield(values; location)
generate_function(cg::CodegenContext{ExecuteRegion}, argtypes, rettypes, reg; name) = reg
function aggregate_funcs(cg::Generate.CodegenContext{ExecuteRegion}, funcs)
    length(funcs) > 1 && error("Functions that are called within an ExecuteRegion context must all be inlined. Found $(length(funcs)-1) explicit calls within.")
    only(funcs)
end

@intrinsic function execute_region(f, T)
    cg = CodegenContext{ExecuteRegion}()
    region = cg(f, Tuple{})
    T(scf.execute_region(;
        region,
        result_0=[IR.Type(T)]
    ) |> IR.result)
end
execute_region(f) = execute_region(f, only(Base.return_types(f, Tuple{}, interp=Generate.MLIRInterpreter())))

abstract type LinalgBody end
generate_return(cg::CodegenContext{LinalgBody}, values; location) = linalg.yield(values; location)
generate_function(cg::CodegenContext{LinalgBody}, argtypes, rettypes, reg; name) = reg
aggregate_funcs(cg::Generate.CodegenContext{LinalgBody}, funcs) = only(funcs)

@intrinsic function _einsum(E::Einsum, Y::T, XS) where {T<:MLIRTensor}
    indexing_maps, iterator_types = maps(E)
    cg = CodegenContext{LinalgBody}()

    region = cg(
        (xs, y)-> execute_region(eltype(T)) do 
            y+prod(xs)
        end,
        Tuple{Tuple{eltype.(XS)...}, eltype(Y)}
    )
    op = linalg.generic(
        XS,
        [Y];
        result_tensors=IR.Type[IR.Type(T)],
        indexing_maps,
        iterator_types,
        region
    )
    return T(IR.result(op))
end

function (E::Einsum)(Y, XS...)
    _einsum(E, Y, XS)
end
