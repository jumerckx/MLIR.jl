module MLIR

using Preferences
using ScopedValues
using MLIR_jll: MLIR_jll

const MLIR_VERSION = ScopedValue(
    VersionNumber(@load_preference("MLIR_VERSION", Base.libllvm_version_string))
)
const MLIR_C_PATH = ScopedValue(@load_preference("MLIR_C_PATH", MLIR_jll.mlir_c))

struct MLIRException <: Exception
    msg::String
end

Base.showerror(io::IO, err::MLIRException) = print(io, err.msg)

include("API/API.jl")
include("IR/IR.jl")
include("Dialects/Dialects.jl")

end # module MLIR
