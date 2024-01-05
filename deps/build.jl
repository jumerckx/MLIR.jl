using LLVM_full_jll

println("Environment")
println("- llvm-config = $(LLVM_full_jll.get_llvm_config_path())")
println("- clang = $(LLVM_full_jll.get_clang_path())")

CXXFLAGS = `$(llvm_config()) --cxxflags` |> readchomp |> split
LDFLAGS = `$(llvm_config()) --ldflags` |> readchomp |> split
println("- CXXFLAGS = $CXXFLAGS")
println("- LDFLAGS = $LDFLAGS")

INCLUDE_PATH = joinpath(LLVM_full_jll.artifact_dir, "include")
DIALECTS_PATH = joinpath(INCLUDE_PATH, "mlir", "Dialect")
println("- INCLUDE_PATH = $INCLUDE_PATH")
println("- DIALECTS_PATH = $DIALECTS_PATH")

# compile TableGen generator
println("Compiling TableGen generator...")
files = [joinpath(@__DIR__, "tblgen", "mlir-jl-tblgen.cc"), joinpath(@__DIR__, "tblgen", "jl-generators.cc")]
output = ["-o", "mlir-jl-tblgen"]
libs = ["-lLLVM", "-lMLIR", "-lMLIRTableGen", "-lLLVMTableGen"]

extra = ["-rpath", joinpath(LLVM_full_jll.artifact_dir, "lib")]
if Base.Sys.isapple()
    isysroot = strip(read(`xcrun --show-sdk-path`, String))
    append!(extra, [
        "-isysroot",
        isysroot,
        "-lc++",
    ])
elseif Base.Sys.islinux()
    append!(extra, [
        "-lstdc++",
    ])
end
println("- extra flags = $extra")

run(`$(clang()) $files $CXXFLAGS $LDFLAGS $extra $libs $output`)

# generate bindings
println("Generating bindings...")

isdir("output") && rm("output"; recursive=true)
mkdir("output")

target_dialects = [
    ("output/Builtin.jl", "../IR/BuiltinOps.td"),
    ("output/AMDGPU.jl", "AMDGPU/AMDGPU.td"),
    ("output/AMX.jl", "AMX/AMX.td"),
    ("output/Affine.jl", "Affine/IR/AffineOps.td"),
    ("output/Arithmetic.jl", "Arithmetic/IR/ArithmeticOps.td"),
    # ("output/ArmNeon.jl", "ArmNeon/ArmNeon.td"),
    ("output/ArmSVE.jl", "ArmSVE/ArmSVE.td"),
    ("output/Async.jl", "Async/IR/AsyncOps.td"),
    ("output/Bufferization.jl", "Bufferization/IR/BufferizationOps.td"),
    ("output/Complex.jl", "Complex/IR/ComplexOps.td"),
    ("output/ControlFlow.jl", "ControlFlow/IR/ControlFlowOps.td"),
    # ("output/DLTI.jl", "DLTI/DLTI.td"),
    ("output/EmitC.jl", "EmitC/IR/EmitC.td"),
    ("output/Func.jl", "Func/IR/FuncOps.td"),
    # ("output/GPU.jl", "GPU/IR/GPUOps.td"),
    ("output/Linalg.jl", "Linalg/IR/LinalgOps.td"),
    # ("output/LinalgStructured.jl", "Linalg/IR/LinalgStructuredOps.td"),
    ("output/LLVMIR.jl", "LLVMIR/LLVMOps.td"),
    # ("output/MLProgram.jl", "MLProgram/IR/MLProgramOps.td"),
    ("output/Math.jl", "Math/IR/MathOps.td"),
    ("output/MemRef.jl", "MemRef/IR/MemRefOps.td"),
    ("output/NVGPU.jl", "NVGPU/IR/NVGPU.td"),
    # ("output/OpenACC.jl", "OpenACC/OpenACCOps.td"),
    # ("output/OpenMP.jl", "OpenMP/OpenMPOps.td"),
    # ("output/PDL.jl", "PDL/IR/PDLOps.td"),
    # ("output/PDLInterp.jl", "PDLInterp/IR/PDLInterpOps.td"),
    ("output/Quant.jl", "Quant/QuantOps.td"),
    # ("output/SCF.jl", "SCF/IR/SCFOps.td"),
    # ("output/SPIRV.jl", "SPIRV/IR/SPIRVOps.td"),
    ("output/Shape.jl", "Shape/IR/ShapeOps.td"),
    ("output/SparseTensor.jl", "SparseTensor/IR/SparseTensorOps.td"),
    ("output/Tensor.jl", "Tensor/IR/TensorOps.td"),
    # ("output/Tosa.jl", "Tosa/IR/TosaOps.td"),
    ("output/Transform.jl", "Transform/IR/TransformOps.td"),
    ("output/Vector.jl", "Vector/IR/VectorOps.td"),
    # ("output/X86Vector.jl", "X86Vector/X86Vector.td"),
]

for (output, path) in target_dialects
    run(`./mlir-jl-tblgen --generator=jl-op-defs $(joinpath(DIALECTS_PATH, path)) -I$INCLUDE_PATH -o $output`)
    println("- Generated \"$output\" from \"$path\"")
end

open(joinpath(@__DIR__, "..", "src", "dialects", "Dialects.jl"), write=true, create=true) do io
    for i in readdir("output")
        mv(joinpath("output", i), joinpath(@__DIR__, "..", "src", "Dialects", i), force=true)
        println(io, "include(\"$i\")")
    end
end