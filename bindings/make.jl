using Pkg
import BinaryBuilderBase: PkgSpec, Prefix, temp_prefix, setup_dependencies, cleanup_dependencies, destdir
using Clang.Generators

function mlir_dialects(version::VersionNumber)
    # construct set of dialects to generate bindings for
    # 1. dialect name
    # 2. bindings file name
    # 3. tablegen files
    dialects = Tuple{String,String,Vector{String}}[
        ("builtin", "Builtin.jl", ["../IR/BuiltinOps.td"]),
    ]

    if version >= v"14"
        append!(dialects, [
            ("amx", "AMX.jl", ["AMX/AMX.td"]),
            ("affine", "Affine.jl", ["Affine/IR/AffineOps.td"]),
            ("arm_neon", "ArmNeon.jl", ["ArmNeon/ArmNeon.td"]),
            ("async", "Async.jl", ["Async/IR/AsyncOps.td"]),
            ("bufferization", "Bufferization.jl", ["Bufferization/IR/BufferizationOps.td"]),
            ("complex", "Complex.jl", ["Complex/IR/ComplexOps.td"]),
            # ("dlti", "DLTI.jl"[, "DLTI/DLTI.td"]), fails on v15
            ("emitc", "EmitC.jl", ["EmitC/IR/EmitC.td"]),
            ("llvm", "LLVMIR.jl", ["LLVMIR/LLVMOps.td"]),
            ("linalg", "Linalg.jl", ["Linalg/IR/LinalgOps.td", "Linalg/IR/LinalgStructuredOps.td"]),
            ("math", "Math.jl", ["Math/IR/MathOps.td"]),
            ("memref", "MemRef.jl", ["MemRef/IR/MemRefOps.td"]),
            ("acc", "OpenACC.jl", ["OpenACC/OpenACCOps.td"]),
            ("omp", "OpenMP.jl", ["OpenMP/OpenMPOps.td"]),
            ("pdl", "PDL.jl", ["PDL/IR/PDLOps.td"]),
            ("pdl_interp", "PDLInterp.jl", ["PDLInterp/IR/PDLInterpOps.td"]),
            ("quant", "Quant.jl", ["Quant/QuantOps.td"]),
            ("shape", "Shape.jl", ["Shape/IR/ShapeOps.td"]),
            ("sparse_tensor", "SparseTensor.jl", ["SparseTensor/IR/SparseTensorOps.td"]),
            ("tensor", "Tensor.jl", ["Tensor/IR/TensorOps.td"]),
            ("tosa", "Tosa.jl", ["Tosa/IR/TosaOps.td"]),
            ("vector", "Vector.jl", ["Vector/IR/VectorOps.td"]),
            ("x86vector", "X86Vector.jl", ["X86Vector/X86Vector.td"]),
        ])
    end

    if v"14" <= version < v"18"
        append!(dialects, [
            ("arm_sve", "ArmSVE.jl", ["ArmSVE/ArmSVE.td"]), # moved to IR subfolder in v18
        ])
    end

    if v"14" <= version < v"15"
        append!(dialects, [
            ("gpu", "GPU.jl", ["GPU/GPUOps.td"]), # moved to IR subfolder in v15
            ("scf", "SCF.jl", ["SCF/SCFOps.td"]), # moved to IR subfolder in v15
            ("std", "StandardOps.jl", ["StandardOps/IR/Ops.td"]),
        ])
    end

    if v"14" <= version < v"16"
        append!(dialects, [
            ("arith", "Arithmetic.jl", ["Arithmetic/IR/ArithmeticOps.td"]), # folder renamed to 'Arith' in v16
            ("spv", "SPIRV.jl", ["SPIRV/IR/SPIRVOps.td"]), # dialect name renamed to 'spirv' in v16
        ])
    end

    if version >= v"15"
        append!(dialects, [
            ("gpu", "GPU.jl", ["GPU/IR/GPUOps.td"]),
            ("scf", "SCF.jl", ["SCF/IR/SCFOps.td"]),
            ("cf", "ControlFlow.jl", ["ControlFlow/IR/ControlFlowOps.td"]),
            ("func", "Func.jl", ["Func/IR/FuncOps.td"]),
            ("ml_program", "MLProgram.jl", ["MLProgram/IR/MLProgramOps.td"]),
            ("nvgpu", "NVGPU.jl", ["NVGPU/IR/NVGPU.td"]),
        ])
    end

    if v"15" <= version < v"18"
        append!(dialects, [
            ("amdgpu", "AMDGPU.jl", ["AMDGPU/AMDGPU.td"]),
        ])
    end

    if v"15" <= version < v"16"
        append!(dialects, [
            ("transform", "Transform.jl", [
                "Transform/IR/TransformOps.td",
                "Bufferization/TransformOps/BufferizationTransformOps.td",
                "Linalg/TransformOps/LinalgTransformOps.td",
                "SCF/TransformOps/SCFTransformOps.td",
            ]), # more ops files in v16
        ])
    end

    if v"16" <= version < v"17"
        append!(dialects, [
            ("transform", "Transform.jl", [
                "Transform/IR/TransformOps.td",
                "Affine/TransformOps/AffineTransformOps.td",
                "Bufferization/TransformOps/BufferizationTransformOps.td",
                "GPU/TransformOps/GPUTransformOps.td",
                "Linalg/TransformOps/LinalgTransformOps.td",
                "MemRef/TransformOps/MemRefTransformOps.td",
                "SCF/TransformOps/SCFTransformOps.td",
                "Vector/TransformOps/VectorTransformOps.td",
            ]), # more ops files in v17
        ])
    end

    if version >= v"16"
        append!(dialects, [
            ("arith", "Arith.jl", ["Arith/IR/ArithOps.td"]),
            ("index", "Index.jl", ["Index/IR/IndexOps.td"]),
            ("spirv", "SPIRV.jl", ["SPIRV/IR/SPIRVOps.td"]),
        ])
    end

    if version >= v"17"
        append!(dialects, [
            # ("arm_sme", "ArmSME.jl", ["ArmSME/IR/ArmSME.td"]),
            # ("irdl", "IRDL.jl", ["IRDL/IR/IRDLOps.td"]),
            # ("ub", "UB.jl", ["UB/IR/UBOps.td"]),
            ("transform", "Transform.jl", [
                "Transform/IR/TransformOps.td",
                "Affine/TransformOps/AffineTransformOps.td",
                "Bufferization/TransformOps/BufferizationTransformOps.td",
                "GPU/TransformOps/GPUTransformOps.td",
                "Linalg/TransformOps/LinalgTransformOps.td",
                "Linalg/TransformOps/LinalgMatchOps.td",
                "MemRef/TransformOps/MemRefTransformOps.td",
                "NVGPU/TransformOps/NVGPUTransformOps.td",
                "SCF/TransformOps/SCFTransformOps.td",
                "Tensor/TransformOps/TensorTransformOps.td",
                "Vector/TransformOps/VectorTransformOps.td",
            ])
        ])
    end

    if version >= v"18"
        append!(dialects, [
            ("mesh", "Mesh.jl", ["Mesh/IR/MeshOps.td"]),
            ("arm_sve", "ArmSVE.jl", ["ArmSVE/IR/ArmSVE.td"]),
            ("amdgpu", "AMDGPU.jl", ["AMDGPU/IR/AMDGPU.td"]),
        ])
    end

    return dialects
end

function rewrite!(dag::ExprDAG) end

julia_llvm = Dict([v"1.11" => v"18"])
options = load_options(joinpath(@__DIR__, "wrap.toml"))

@add_def off_t
@add_def MlirTypesCallback

for (julia_version, llvm_version) in julia_llvm
    println("Generating... julia version: $julia_version, llvm version: $llvm_version")

    temp_prefix() do prefix
        mlir_jl_tblgen = "/home/jumerckx/.julia/scratchspaces/bfde9dd4-8f40-4a1e-be09-1475335e1c92/build/bin/mlir-jl-tblgen"
        include_dir = "/home/jumerckx/masterthesis/llvm-project/llvm/install/relwithdebinfo/include"

        # generate MLIR API bindings
        mkpath(joinpath(@__DIR__, "..", "src", "API", string(llvm_version.major)))

        let options = deepcopy(options)
            output_file_path = joinpath(@__DIR__, "..", "src", "API", string(llvm_version.major), options["general"]["output_file_path"])
            isdir(dirname(output_file_path)) || mkpath(dirname(output_file_path))
            options["general"]["output_file_path"] = output_file_path

            libmlir_header_dir = joinpath(include_dir, "mlir-c")
            args = Generators.get_default_args()
            append!(args, ["-I", include_dir, "-x", "c++"])

            headers = detect_headers(libmlir_header_dir, args, Dict(), endswith("Python/Interop.h"))
            ctx = create_context(headers, args, options)

            # build without printing so we can do custom rewriting
            build!(ctx, BUILDSTAGE_NO_PRINTING)

            rewrite!(ctx.dag)

            # print
            build!(ctx, BUILDSTAGE_PRINTING_ONLY)
        end

        # generate MLIR dialect bindings
        mkpath(joinpath(@__DIR__, "..", "src", "Dialects", string(llvm_version.major)))

        for (dialect_name, binding, tds) in mlir_dialects(llvm_version)
            tempfiles = map(tds) do td
                tempfile, _ = mktemp()
                flags = [
                    "--generator=jl-op-defs",
                    "--disable-module-wrap",
                    joinpath(include_dir, "mlir", "Dialect", td),
                    "-I", include_dir,
                    "-o", tempfile,
                ]
                run(`$mlir_jl_tblgen $flags`)
                return tempfile
            end

            output = joinpath(@__DIR__, "..", "src", "Dialects", string(llvm_version.major), binding)
            open(output, "w") do io
                println(io, "module $dialect_name\n")
                for tempfile in tempfiles
                    write(io, read(tempfile, String))
                end
                println(io, "end # $dialect_name")
            end

            println("- Generated \"$binding\" from $(join(tds, ",", " and "))")
        end
    end
end