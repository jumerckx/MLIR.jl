LLVM_DIR = /home/jumerckx/masterthesis/llvm-project/llvm
llvmconfig = $(LLVM_DIR)/install_debug/bin/llvm-config

CXXFLAGS = $(shell $(llvmconfig) --cxxflags)
LDFLAGS = $(shell $(llvmconfig) --ldflags)

mlir-jl-tblgen: mlir-jl-tblgen.cc jl-generators-2.cc
	$(CXX) -g $^ $(CXXFLAGS) $(LDFLAGS) -I$(LLVM_DIR)/include -L$(LLVM_DIR)/lib -lMLIR -lLLVM -lMLIRTableGen -lLLVMTableGen -o $@ 
# mlir-jl-tblgen: mlir-jl-tblgen.cc jl-generators.cc
# 	$(CXX) -g $^ $(CXXFLAGS) $(LDFLAGS) -I$(LLVM_DIR)/include -L$(LLVM_DIR)/lib -lMLIR -lLLVM -lMLIRTableGen -lLLVMTableGen -o $@ 

# LD_LIBRARY_PATH=/home/jumerckx/masterthesis/llvm-project/llvm/build_debug/lib ./mlir-jl-tblgen --explain-missing --generator=jl-op-defs /home/jumerckx/masterthesis/llvm-project/llvm/install_debug/include/mlir/Dialect/Arith/IR/ArithOps.td -I/home/jumerckx/masterthesis/llvm-project/llvm/install_debug/include --strip-prefix=Arith_ > output/Arith.jl

# LD_LIBRARY_PATH=/home/jumerckx/masterthesis/llvm-project/llvm/build_debug/lib ./mlir-jl-tblgen --explain-missing --generator=jl-op-defs /home/jumerckx/masterthesis/llvm-project/llvm/install_debug/include/mlir/Dialect/LLVMIR/LLVMOps.td -I/home/jumerckx/masterthesis/llvm-project/llvm/install_debug/include --strip-prefix=LLVM_ > output/Llvm.jl

# LD_LIBRARY_PATH=/home/jumerckx/masterthesis/llvm-project/llvm/build_debug/lib ./mlir-jl-tblgen --explain-missing --generator=jl-op-defs /home/jumerckx/masterthesis/llvm-project/llvm/install_debug/include/mlir/Dialect/ControlFlow/IR/ControlFlowOps.td -I/home/jumerckx/masterthesis/llvm-project/llvm/install_debug/include > output/Controlflow.jl

# LD_LIBRARY_PATH=/home/jumerckx/masterthesis/llvm-project/llvm/build_debug/lib ./mlir-jl-tblgen --explain-missing --generator=jl-op-defs /home/jumerckx/masterthesis/llvm-project/llvm/install_debug/include/mlir/Dialect/Func/IR/FuncOps.td -I/home/jumerckx/masterthesis/llvm-project/llvm/install_debug/include > output/Func.jl

# LD_LIBRARY_PATH=/home/jumerckx/masterthesis/llvm-project/llvm/build_debug/lib ./mlir-jl-tblgen --explain-missing --generator=jl-op-defs /home/jumerckx/masterthesis/llvm-project/llvm/install_debug/include/mlir/Dialect/GPU/IR/GPUOps.td -I/home/jumerckx/masterthesis/llvm-project/llvm/install_debug/include --strip-prefix=GPU_ > output/Gpu.jl

# LD_LIBRARY_PATH=/home/jumerckx/masterthesis/llvm-project/llvm/build_debug/lib ./mlir-jl-tblgen --explain-missing --generator=jl-op-defs /home/jumerckx/masterthesis/llvm-project/llvm/install_debug/include/mlir/Dialect/PDL/IR/PDLOps.td -I/home/jumerckx/masterthesis/llvm-project/llvm/install_debug/include --strip-prefix=PDL_ > output/Pdl.jl

# LD_LIBRARY_PATH=/home/jumerckx/masterthesis/llvm-project/llvm/build_debug/lib ./mlir-jl-tblgen --explain-missing --generator=jl-op-defs /home/jumerckx/masterthesis/llvm-project/llvm/install_debug/include/mlir/IR/BuiltinOps.td -I/home/jumerckx/masterthesis/llvm-project/llvm/install_debug/include  > output/Builtin.jl

# LD_LIBRARY_PATH=/home/jumerckx/masterthesis/llvm-project/llvm/build_debug/lib ./mlir-jl-tblgen --explain-missing --generator=jl-op-defs /home/jumerckx/masterthesis/llvm-project/llvm/install_debug/include/mlir/IR/BuiltinOps.td -I/home/jumerckx/masterthesis/llvm-project/llvm/install_debug/include  > output/Scf.jl