LLVM_DIR = /home/jumerckx/masterthesis/llvm-project/llvm
llvmconfig = /home/jumerckx/masterthesis/llvm-project/build/llvm_install/bin/llvm-config

CXXFLAGS = $(shell $(llvmconfig) --cxxflags)
LDFLAGS = $(shell $(llvmconfig) --ldflags)

mlir-jl-tblgen: mlir-jl-tblgen.cc jl-generators.cc
	$(CXX) -g $^ $(CXXFLAGS) $(LDFLAGS) -I$(LLVM_DIR)/include -L$(LLVM_DIR)/lib -lMLIR -lLLVM -lMLIRTableGen -lLLVMTableGen -o $@ 

# LD_LIBRARY_PATH=llvm/lib ./mlir-jl-tblgen --generator=jl-op-defs llvm/include/mlir/Dialect/StandardOps/IR/Ops.td -Illvm/include