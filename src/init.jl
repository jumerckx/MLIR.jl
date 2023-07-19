function __init__()
    ccall((:brutus_init, MLIR.API.mlir_c), Cvoid, (Any,), @__MODULE__)
end
