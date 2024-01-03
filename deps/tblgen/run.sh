#! /usr/bin/env bash
set -e
cd $(dirname "$0")

make mlir-jl-tblgen

rm -f ./output/*.jl
mkdir -p ./output

LD_LIBRARY_PATH=/home/jumerckx/masterthesis/llvm-project/build/llvm_build/lib/
# DIALECTS_PATH=/home/jumerckx/masterthesis/llvm-project/llvm/install_debug/include/mlir/Dialect/
INCLUDE_PATH=/home/jumerckx/masterthesis/llvm-project/build/llvm_install/include/

# LD_LIBRARY_PATH=/home/jumerckx/.julia/artifacts/7a30d5d08131c8d72e002314ee933895a1bed594/mlir/lib/
# INCLUDE_PATH=/home/jumerckx/.julia/artifacts/7a30d5d08131c8d72e002314ee933895a1bed594/mlir/include/
DIALECTS_PATH=/home/jumerckx/masterthesis/llvm-project/mlir/include/mlir/Dialect/

LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/Arith/IR/ArithOps.td -I$INCLUDE_PATH > output/arith.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/Index/IR/IndexOps.td -I$INCLUDE_PATH > output/index.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/Shape/IR/ShapeOps.td -I$INCLUDE_PATH > output/shape.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/MemRef/IR/MemRefOps.td -I$INCLUDE_PATH > output/memref.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/Complex/IR/ComplexOps.td -I$INCLUDE_PATH > output/complex.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/Linalg/IR/LinalgOps.td -I$INCLUDE_PATH > output/linalg.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/Linalg/IR/LinalgStructuredOps.td -I$INCLUDE_PATH > output/linalg_structured.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/LLVMIR/LLVMOps.td -I$INCLUDE_PATH > output/llvm.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/AMDGPU/IR/AMDGPU.td -I$INCLUDE_PATH > output/amdgpu.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/Affine/IR/AffineOps.td -I$INCLUDE_PATH > output/affine.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/ControlFlow/IR/ControlFlowOps.td -I$INCLUDE_PATH > output/cf.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/Func/IR/FuncOps.td -I$INCLUDE_PATH > output/func.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/Quant/QuantOps.td -I$INCLUDE_PATH > output/quant.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/Transform/IR/TransformOps.td -I$INCLUDE_PATH > output/transform.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/Linalg/TransformOps/LinalgTransformOps.td -I$INCLUDE_PATH > output/transform_linalg.jl
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/IRDL/IR/IRDLOps.td -I$INCLUDE_PATH > output/IRDL.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs $DIALECTS_PATH/PDL/IR/PDLOps.td -I$INCLUDE_PATH > output/pdl.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs /home/jumerckx/masterthesis/llvm-project/mlir/include/mlir/IR/BuiltinOps.td --dialect-name="builtin"  -I$INCLUDE_PATH > output/builtin.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs /home/jumerckx/masterthesis/llvm-project/mlir/include/mlir/IR/BuiltinTypes.td --dialect-name="builtin" -I$INCLUDE_PATH > output/builtin_types.jl
LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./mlir-jl-tblgen --generator=jl-op-defs /home/jumerckx/masterthesis/llvm-project/mlir/include/mlir/IR/OpBase.td --dialect-name="base" -I$INCLUDE_PATH > output/base.jl
