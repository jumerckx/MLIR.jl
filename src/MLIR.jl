module MLIR

include("get_artifact.jl")

import LLVM

module API
    using CEnum

    # MLIR C API
    import ..LLVM

    using Pkg.Artifacts
    MLIRdist_path = artifact"MLIRdist"
    const mlir_c = joinpath(MLIRdist_path, "mlir/lib/libMLIR-C.so")
    const mlir_runner_utils = joinpath(MLIRdist_path, "mlir/lib/libmlir_runner_utils.so")
    const mlir_c_runner_utils = joinpath(MLIRdist_path, "mlir/lib/libmlir_c_runner_utils.so")

    let
        ver = "18"
        dir = joinpath(@__DIR__, "..", "lib", ver)
        if !isdir(dir)
            error("""The MLIR API bindings for v$ver do not exist.
                    You might need a newer version of MLIR.jl for this version of Julia.""")
        end

        include(joinpath(dir, "libMLIR_h.jl"))
    end
end # module API

# MlirStringRef is a non-owning reference to a string,
# we thus need to ensure that the Julia string remains alive
# over the use. For that we use the cconvert/unsafe_convert mechanism
# for foreign-calls. The returned value of the cconvert is rooted across
# foreign-call.
Base.cconvert(::Type{API.MlirStringRef}, s::Union{Symbol, String}) = s
Base.cconvert(::Type{API.MlirStringRef}, s::AbstractString) =
    Base.cconvert(API.MlirStringRef, String(s)::String)

# Directly create `MlirStringRef` instead of adding an extra ccall.
function Base.unsafe_convert(::Type{API.MlirStringRef}, s::Union{Symbol, String, AbstractVector{UInt8}})
    p = Base.unsafe_convert(Ptr{Cchar}, s)
    return API.MlirStringRef(p, length(s))
end

module IR
    import ..API: API

    include("./IR/IR.jl")
    include("./IR/state.jl")
end # module IR

include("./Dialects.jl")
include("./Affine.jl")

end # module MLIR
