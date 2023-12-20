module Llvm

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation, context
import ...API

make_named_attribute(name, val) = make_named_attribute(name, Attribute(val))

make_named_attribute(name, val::Attribute) = NamedAttribute(name, val)

function make_named_attribute(name, val::NamedAttribute)
  assert(true) # TODO(jm): check whether name of attribute is correct, getting the name might need to be added to IR.jl?
  return val
end

"""
ashr

"""
function AShr(; location::Location, res_::MLIRType, lhs_::Value, rhs_::Value)
  results = [res_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.ashr", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
add

"""
function Add(; location::Location, res_::MLIRType, lhs_::Value, rhs_::Value)
  results = [res_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.add", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
addrspacecast

"""
function AddrSpaceCast(; location::Location, res_::MLIRType, arg_::Value)
  results = [res_]
  operands = [arg_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.addrspacecast", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
mlir.addressof

Creates an SSA value containing a pointer to a global variable or constant
defined by `llvm.mlir.global`. The global value can be defined after its
first referenced. If the global value is a constant, storing into it is not
allowed.

Examples:

```mlir
func @foo() {
  // Get the address of a global variable.
  %0 = llvm.mlir.addressof @const : !llvm.ptr<i32>

  // Use it as a regular pointer.
  %1 = llvm.load %0 : !llvm.ptr<i32>

  // Get the address of a function.
  %2 = llvm.mlir.addressof @foo : !llvm.ptr<func<void ()>>

  // The function address can be used for indirect calls.
  llvm.call %2() : () -> ()
}

// Define the global.
llvm.mlir.global @const(42 : i32) : i32
```
  
"""
function AddressOf(; location::Location, res_::MLIRType, global_name_::Union{NamedAttribute, Attribute})
  results = [res_]
  operands = []
  regions = []
  successors = []
  attributes = [make_named_attribute("global_name", global_name_)]
  
  create_operation(
        "llvm.mlir.addressof", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
alloca

"""
function Alloca(; location::Location, res_::MLIRType, arraySize_::Value, alignment_=nothing::Union{Nothing, Union{NamedAttribute, Int64}}, elem_type_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, inalloca_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}})
  results = [res_]
  operands = [arraySize_]
  regions = []
  successors = []
  attributes = []

  (alignment_ != nothing) && push!(attributes, make_named_attribute("alignment", alignment_))
  (elem_type_ != nothing) && push!(attributes, make_named_attribute("elem_type", elem_type_))
  (inalloca_ != nothing) && push!(attributes, make_named_attribute("inalloca", inalloca_))

  create_operation(
        "llvm.alloca", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
and

"""
function And(; location::Location, res_::MLIRType, lhs_::Value, rhs_::Value)
  results = [res_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.and", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
cmpxchg

"""
function AtomicCmpXchg(; location::Location, res_::MLIRType, ptr_::Value, cmp_::Value, val_::Value, success_ordering_::Union{NamedAttribute, Attribute}, failure_ordering_::Union{NamedAttribute, Attribute}, syncscope_=nothing::Union{Nothing, Union{NamedAttribute, String}}, alignment_=nothing::Union{Nothing, Union{NamedAttribute, Int64}}, weak_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, volatile__=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, access_groups_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, alias_scopes_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, noalias_scopes_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, tbaa_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}})
  results = [res_]
  operands = [ptr_, cmp_, val_]
  regions = []
  successors = []
  attributes = [make_named_attribute("success_ordering", success_ordering_), make_named_attribute("failure_ordering", failure_ordering_)]

  (syncscope_ != nothing) && push!(attributes, make_named_attribute("syncscope", syncscope_))
  (alignment_ != nothing) && push!(attributes, make_named_attribute("alignment", alignment_))
  (weak_ != nothing) && push!(attributes, make_named_attribute("weak", weak_))
  (volatile__ != nothing) && push!(attributes, make_named_attribute("volatile_", volatile__))
  (access_groups_ != nothing) && push!(attributes, make_named_attribute("access_groups", access_groups_))
  (alias_scopes_ != nothing) && push!(attributes, make_named_attribute("alias_scopes", alias_scopes_))
  (noalias_scopes_ != nothing) && push!(attributes, make_named_attribute("noalias_scopes", noalias_scopes_))
  (tbaa_ != nothing) && push!(attributes, make_named_attribute("tbaa", tbaa_))

  create_operation(
        "llvm.cmpxchg", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
atomicrmw

"""
function AtomicRMW(; location::Location, res_::MLIRType, ptr_::Value, val_::Value, bin_op_::Union{NamedAttribute, Attribute}, ordering_::Union{NamedAttribute, Attribute}, syncscope_=nothing::Union{Nothing, Union{NamedAttribute, String}}, alignment_=nothing::Union{Nothing, Union{NamedAttribute, Int64}}, volatile__=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, access_groups_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, alias_scopes_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, noalias_scopes_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, tbaa_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}})
  results = [res_]
  operands = [ptr_, val_]
  regions = []
  successors = []
  attributes = [make_named_attribute("bin_op", bin_op_), make_named_attribute("ordering", ordering_)]

  (syncscope_ != nothing) && push!(attributes, make_named_attribute("syncscope", syncscope_))
  (alignment_ != nothing) && push!(attributes, make_named_attribute("alignment", alignment_))
  (volatile__ != nothing) && push!(attributes, make_named_attribute("volatile_", volatile__))
  (access_groups_ != nothing) && push!(attributes, make_named_attribute("access_groups", access_groups_))
  (alias_scopes_ != nothing) && push!(attributes, make_named_attribute("alias_scopes", alias_scopes_))
  (noalias_scopes_ != nothing) && push!(attributes, make_named_attribute("noalias_scopes", noalias_scopes_))
  (tbaa_ != nothing) && push!(attributes, make_named_attribute("tbaa", tbaa_))

  create_operation(
        "llvm.atomicrmw", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
bitcast

"""
function Bitcast(; location::Location, res_::MLIRType, arg_::Value)
  results = [res_]
  operands = [arg_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.bitcast", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
br

"""
function Br(; location::Location, destOperands_::Vector{Value}, dest_::Block, loop_annotation_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}})
  results = []
  operands = [destOperands_...]
  regions = []
  successors = [dest_]
  attributes = []

  (loop_annotation_ != nothing) && push!(attributes, make_named_attribute("loop_annotation", loop_annotation_))

  create_operation(
        "llvm.br", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
call_intrinsic

Call the specified llvm intrinsic. If the intrinsic is overloaded, use
the MLIR function type of this op to determine which intrinsic to call.
  
"""
function CallIntrinsic(; location::Location, results_=nothing::Union{Nothing, MLIRType}, args_::Vector{Value}, intrin_::Union{NamedAttribute, String}, fastmathFlags_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}})
  results = []
  (results_ != nothing) && push!(results, results_)
  operands = [args_...]
  regions = []
  successors = []
  attributes = [make_named_attribute("intrin", intrin_)]

  (fastmathFlags_ != nothing) && push!(attributes, make_named_attribute("fastmathFlags", fastmathFlags_))

  create_operation(
        "llvm.call_intrinsic", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
call

In LLVM IR, functions may return either 0 or 1 value. LLVM IR dialect
implements this behavior by providing a variadic `call` operation for 0- and
1-result functions. Even though MLIR supports multi-result functions, LLVM
IR dialect disallows them.

The `call` instruction supports both direct and indirect calls. Direct calls
start with a function name (`@`-prefixed) and indirect calls start with an
SSA value (`%`-prefixed). The direct callee, if present, is stored as a
function attribute `callee`. The trailing type list contains the optional
indirect callee type and the MLIR function type, which differs from the
LLVM function type that uses a explicit void type to model functions that do
not return a value.

Examples:

```mlir
// Direct call without arguments and with one result.
%0 = llvm.call @foo() : () -> (f32)

// Direct call with arguments and without a result.
llvm.call @bar(%0) : (f32) -> ()

// Indirect call with an argument and without a result.
llvm.call %1(%0) : !llvm.ptr, (f32) -> ()
```
  
"""
function Call(; location::Location, result_=nothing::Union{Nothing, MLIRType}, _unnamed0_::Vector{Value}, callee_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, fastmathFlags_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, branch_weights_=nothing::Union{Nothing, Union{NamedAttribute, Vector{Float32}}}, access_groups_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, alias_scopes_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, noalias_scopes_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, tbaa_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}})
  results = []
  (result_ != nothing) && push!(results, result_)
  operands = [_unnamed0_...]
  regions = []
  successors = []
  attributes = []

  (callee_ != nothing) && push!(attributes, make_named_attribute("callee", callee_))
  (fastmathFlags_ != nothing) && push!(attributes, make_named_attribute("fastmathFlags", fastmathFlags_))
  (branch_weights_ != nothing) && push!(attributes, make_named_attribute("branch_weights", branch_weights_))
  (access_groups_ != nothing) && push!(attributes, make_named_attribute("access_groups", access_groups_))
  (alias_scopes_ != nothing) && push!(attributes, make_named_attribute("alias_scopes", alias_scopes_))
  (noalias_scopes_ != nothing) && push!(attributes, make_named_attribute("noalias_scopes", noalias_scopes_))
  (tbaa_ != nothing) && push!(attributes, make_named_attribute("tbaa", tbaa_))

  create_operation(
        "llvm.call", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
comdat

Provides access to object file COMDAT section/group functionality.

Examples:
```mlir
llvm.comdat @__llvm_comdat {
  llvm.comdat_selector @any any
}
llvm.mlir.global internal constant @has_any_comdat(1 : i64) comdat(@__llvm_comdat::@any) : i64
```
  
"""
function Comdat(; location::Location, body_::Region, sym_name_::Union{NamedAttribute, Attribute})
  results = []
  operands = []
  regions = [body_]
  successors = []
  attributes = [make_named_attribute("sym_name", sym_name_)]
  
  create_operation(
        "llvm.comdat", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
comdat_selector

Provides access to object file COMDAT section/group functionality.

Examples:
```mlir
llvm.comdat @__llvm_comdat {
  llvm.comdat_selector @any any
}
llvm.mlir.global internal constant @has_any_comdat(1 : i64) comdat(@__llvm_comdat::@any) : i64
```
  
"""
function ComdatSelector(; location::Location, sym_name_::Union{NamedAttribute, Attribute}, comdat_::Union{NamedAttribute, Attribute})
  results = []
  operands = []
  regions = []
  successors = []
  attributes = [make_named_attribute("sym_name", sym_name_), make_named_attribute("comdat", comdat_)]
  
  create_operation(
        "llvm.comdat_selector", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
cond_br

"""
function CondBr(; location::Location, condition_::Value, trueDestOperands_::Vector{Value}, falseDestOperands_::Vector{Value}, trueDest_::Block, falseDest_::Block, branch_weights_=nothing::Union{Nothing, Union{NamedAttribute, Vector{Float32}}}, loop_annotation_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}})
  results = []
  operands = [condition_, trueDestOperands_..., falseDestOperands_...]
  regions = []
  successors = [trueDest_, falseDest_]
  attributes = []
  # attributes = [make_named_attribute("operand_segment_sizes", Attribute(API.mlirDenseI32ArrayGet(
  #   context().context,
  #   1 + length(trueDestOperands_) + length(falseDestOperands_),
  #   Int32[1, length(trueDestOperands_), length(falseDestOperands_)])))]

  (branch_weights_ != nothing) && push!(attributes, make_named_attribute("branch_weights", branch_weights_))
  (loop_annotation_ != nothing) && push!(attributes, make_named_attribute("loop_annotation", loop_annotation_))

  push!(attributes, make_named_attribute("operand_segment_sizes", Attribute(API.mlirDenseI32ArrayGet(
    context().context,
    1 + length(trueDestOperands_) + length(falseDestOperands_),
    Int32[1, length(trueDestOperands_), length(falseDestOperands_)]))))
  # push!(attributes, make_named_attribute("operand_segment_sizes", Int32[1, length(trueDestOperands_), length(falseDestOperands_)]))
  
  create_operation(
        "llvm.cond_br", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
mlir.constant

Unlike LLVM IR, MLIR does not have first-class constant values. Therefore,
all constants must be created as SSA values before being used in other
operations. `llvm.mlir.constant` creates such values for scalars and
vectors. It has a mandatory `value` attribute, which may be an integer,
floating point attribute; dense or sparse attribute containing integers or
floats. The type of the attribute is one of the corresponding MLIR builtin
types. It may be omitted for `i64` and `f64` types that are implied. The
operation produces a new SSA value of the specified LLVM IR dialect type.
The type of that value _must_ correspond to the attribute type converted to
LLVM IR.

Examples:

```mlir
// Integer constant, internal i32 is mandatory
%0 = llvm.mlir.constant(42 : i32) : i32

// It\'s okay to omit i64.
%1 = llvm.mlir.constant(42) : i64

// Floating point constant.
%2 = llvm.mlir.constant(42.0 : f32) : f32

// Splat dense vector constant.
%3 = llvm.mlir.constant(dense<1.0> : vector<4xf32>) : vector<4xf32>
```
  
"""
function Constant(; location::Location, res_::MLIRType, value_::Union{NamedAttribute, Attribute})
  results = [res_]
  operands = []
  regions = []
  successors = []
  attributes = [make_named_attribute("value", value_)]
  
  create_operation(
        "llvm.mlir.constant", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
extractelement

"""
function ExtractElement(; location::Location, res_::MLIRType, vector_::Value, position_::Value)
  results = [res_]
  operands = [vector_, position_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.extractelement", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
extractvalue

"""
function ExtractValue(; location::Location, res_::MLIRType, container_::Value, position_::Union{NamedAttribute, Attribute})
  results = [res_]
  operands = [container_]
  regions = []
  successors = []
  attributes = [make_named_attribute("position", position_)]
  
  create_operation(
        "llvm.extractvalue", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
fadd

"""
function FAdd(; location::Location, res_::MLIRType, lhs_::Value, rhs_::Value, fastmathFlags_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}})
  results = [res_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []

  (fastmathFlags_ != nothing) && push!(attributes, make_named_attribute("fastmathFlags", fastmathFlags_))

  create_operation(
        "llvm.fadd", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
fcmp

"""
function FCmp(; location::Location, res_::MLIRType, lhs_::Value, rhs_::Value, predicate_::Union{NamedAttribute, Attribute}, fastmathFlags_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}})
  results = [res_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = [make_named_attribute("predicate", predicate_)]

  (fastmathFlags_ != nothing) && push!(attributes, make_named_attribute("fastmathFlags", fastmathFlags_))

  create_operation(
        "llvm.fcmp", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
fdiv

"""
function FDiv(; location::Location, res_::MLIRType, lhs_::Value, rhs_::Value, fastmathFlags_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}})
  results = [res_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []

  (fastmathFlags_ != nothing) && push!(attributes, make_named_attribute("fastmathFlags", fastmathFlags_))

  create_operation(
        "llvm.fdiv", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
fmul

"""
function FMul(; location::Location, res_::MLIRType, lhs_::Value, rhs_::Value, fastmathFlags_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}})
  results = [res_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []

  (fastmathFlags_ != nothing) && push!(attributes, make_named_attribute("fastmathFlags", fastmathFlags_))

  create_operation(
        "llvm.fmul", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
fneg

"""
function FNeg(; location::Location, res_::MLIRType, operand_::Value, fastmathFlags_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}})
  results = [res_]
  operands = [operand_]
  regions = []
  successors = []
  attributes = []

  (fastmathFlags_ != nothing) && push!(attributes, make_named_attribute("fastmathFlags", fastmathFlags_))

  create_operation(
        "llvm.fneg", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
fpext

"""
function FPExt(; location::Location, res_::MLIRType, arg_::Value)
  results = [res_]
  operands = [arg_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.fpext", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
fptosi

"""
function FPToSI(; location::Location, res_::MLIRType, arg_::Value)
  results = [res_]
  operands = [arg_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.fptosi", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
fptoui

"""
function FPToUI(; location::Location, res_::MLIRType, arg_::Value)
  results = [res_]
  operands = [arg_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.fptoui", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
fptrunc

"""
function FPTrunc(; location::Location, res_::MLIRType, arg_::Value)
  results = [res_]
  operands = [arg_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.fptrunc", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
frem

"""
function FRem(; location::Location, res_::MLIRType, lhs_::Value, rhs_::Value, fastmathFlags_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}})
  results = [res_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []

  (fastmathFlags_ != nothing) && push!(attributes, make_named_attribute("fastmathFlags", fastmathFlags_))

  create_operation(
        "llvm.frem", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
fsub

"""
function FSub(; location::Location, res_::MLIRType, lhs_::Value, rhs_::Value, fastmathFlags_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}})
  results = [res_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []

  (fastmathFlags_ != nothing) && push!(attributes, make_named_attribute("fastmathFlags", fastmathFlags_))

  create_operation(
        "llvm.fsub", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
fence

"""
function Fence(; location::Location, ordering_::Union{NamedAttribute, Attribute}, syncscope_=nothing::Union{Nothing, Union{NamedAttribute, String}})
  results = []
  operands = []
  regions = []
  successors = []
  attributes = [make_named_attribute("ordering", ordering_)]

  (syncscope_ != nothing) && push!(attributes, make_named_attribute("syncscope", syncscope_))

  create_operation(
        "llvm.fence", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
freeze

"""
function Freeze(; location::Location, res_::MLIRType, val_::Value)
  results = [res_]
  operands = [val_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.freeze", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
getelementptr

This operation mirrors LLVM IRs \'getelementptr\' operation that is used to
perform pointer arithmetic.

Like in LLVM IR, it is possible to use both constants as well as SSA values
as indices. In the case of indexing within a structure, it is required to
either use constant indices directly, or supply a constant SSA value.

An optional \'inbounds\' attribute specifies the low-level pointer arithmetic
overflow behavior that LLVM uses after lowering the operation to LLVM IR.

Examples:

```mlir
// GEP with an SSA value offset
%0 = llvm.getelementptr %1[%2] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>

// GEP with a constant offset and the inbounds attribute set
%0 = llvm.getelementptr inbounds %1[3] : (!llvm.ptr<f32>) -> !llvm.ptr<f32>

// GEP with constant offsets into a structure
%0 = llvm.getelementptr %1[0, 1]
   : (!llvm.ptr<struct(i32, f32)>) -> !llvm.ptr<f32>
```
  
"""
function GEP(; location::Location, res_::MLIRType, base_::Value, dynamicIndices_::Vector{Value}, rawConstantIndices_::Union{NamedAttribute, Vector{Float32}}, elem_type_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, inbounds_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}})
  results = [res_]
  operands = [base_, dynamicIndices_...]
  regions = []
  successors = []
  attributes = [make_named_attribute("rawConstantIndices", rawConstantIndices_)]

  (elem_type_ != nothing) && push!(attributes, make_named_attribute("elem_type", elem_type_))
  (inbounds_ != nothing) && push!(attributes, make_named_attribute("inbounds", inbounds_))

  create_operation(
        "llvm.getelementptr", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
mlir.global_ctors

Specifies a list of constructor functions and priorities. The functions
referenced by this array will be called in ascending order of priority (i.e.
lowest first) when the module is loaded. The order of functions with the
same priority is not defined. This operation is translated to LLVM\'s
global_ctors global variable. The initializer functions are run at load
time. The `data` field present in LLVM\'s global_ctors variable is not
modeled here.

Examples:

```mlir
llvm.mlir.global_ctors {@ctor}

llvm.func @ctor() {
  ...
  llvm.return
}
```

  
"""
function GlobalCtors(; location::Location, ctors_::Union{NamedAttribute, Attribute}, priorities_::Union{NamedAttribute, Attribute})
  results = []
  operands = []
  regions = []
  successors = []
  attributes = [make_named_attribute("ctors", ctors_), make_named_attribute("priorities", priorities_)]
  
  create_operation(
        "llvm.mlir.global_ctors", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
mlir.global_dtors

Specifies a list of destructor functions and priorities. The functions
referenced by this array will be called in descending order of priority (i.e.
highest first) when the module is unloaded. The order of functions with the
same priority is not defined. This operation is translated to LLVM\'s
global_dtors global variable. The `data` field present in LLVM\'s
global_dtors variable is not modeled here.

Examples:

```mlir
llvm.func @dtor() {
  llvm.return
}
llvm.mlir.global_dtors {@dtor}
```

  
"""
function GlobalDtors(; location::Location, dtors_::Union{NamedAttribute, Attribute}, priorities_::Union{NamedAttribute, Attribute})
  results = []
  operands = []
  regions = []
  successors = []
  attributes = [make_named_attribute("dtors", dtors_), make_named_attribute("priorities", priorities_)]
  
  create_operation(
        "llvm.mlir.global_dtors", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
mlir.global

Since MLIR allows for arbitrary operations to be present at the top level,
global variables are defined using the `llvm.mlir.global` operation. Both
global constants and variables can be defined, and the value may also be
initialized in both cases.

There are two forms of initialization syntax. Simple constants that can be
represented as MLIR attributes can be given in-line:

```mlir
llvm.mlir.global @variable(32.0 : f32) : f32
```

This initialization and type syntax is similar to `llvm.mlir.constant` and
may use two types: one for MLIR attribute and another for the LLVM value.
These types must be compatible.

More complex constants that cannot be represented as MLIR attributes can be
given in an initializer region:

```mlir
// This global is initialized with the equivalent of:
//   i32* getelementptr (i32* @g2, i32 2)
llvm.mlir.global constant @int_gep() : !llvm.ptr<i32> {
  %0 = llvm.mlir.addressof @g2 : !llvm.ptr<i32>
  %1 = llvm.mlir.constant(2 : i32) : i32
  %2 = llvm.getelementptr %0[%1]
     : (!llvm.ptr<i32>, i32) -> !llvm.ptr<i32>
  // The initializer region must end with `llvm.return`.
  llvm.return %2 : !llvm.ptr<i32>
}
```

Only one of the initializer attribute or initializer region may be provided.

`llvm.mlir.global` must appear at top-level of the enclosing module. It uses
an @-identifier for its value, which will be uniqued by the module with
respect to other @-identifiers in it.

Examples:

```mlir
// Global values use @-identifiers.
llvm.mlir.global constant @cst(42 : i32) : i32

// Non-constant values must also be initialized.
llvm.mlir.global @variable(32.0 : f32) : f32

// Strings are expected to be of wrapped LLVM i8 array type and do not
// automatically include the trailing zero.
llvm.mlir.global @string(\"abc\") : !llvm.array<3 x i8>

// For strings globals, the trailing type may be omitted.
llvm.mlir.global constant @no_trailing_type(\"foo bar\")

// A complex initializer is constructed with an initializer region.
llvm.mlir.global constant @int_gep() : !llvm.ptr<i32> {
  %0 = llvm.mlir.addressof @g2 : !llvm.ptr<i32>
  %1 = llvm.mlir.constant(2 : i32) : i32
  %2 = llvm.getelementptr %0[%1]
     : (!llvm.ptr<i32>, i32) -> !llvm.ptr<i32>
  llvm.return %2 : !llvm.ptr<i32>
}
```

Similarly to functions, globals have a linkage attribute. In the custom
syntax, this attribute is placed between `llvm.mlir.global` and the optional
`constant` keyword. If the attribute is omitted, `external` linkage is
assumed by default.

Examples:

```mlir
// A constant with internal linkage will not participate in linking.
llvm.mlir.global internal constant @cst(42 : i32) : i32

// By default, \"external\" linkage is assumed and the global participates in
// symbol resolution at link-time.
llvm.mlir.global @glob(0 : f32) : f32

// Alignment is optional
llvm.mlir.global private constant @y(dense<1.0> : tensor<8xf32>) : !llvm.array<8 x f32>
```

Like global variables in LLVM IR, globals can have an (optional)
alignment attribute using keyword `alignment`. The integer value of the
alignment must be a positive integer that is a power of 2.

Examples:

```mlir
// Alignment is optional
llvm.mlir.global private constant @y(dense<1.0> : tensor<8xf32>) { alignment = 32 : i64 } : !llvm.array<8 x f32>
```

  
"""
function Global(; location::Location, initializer_::Region, global_type_::Union{NamedAttribute, Attribute}, constant_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, sym_name_::Union{NamedAttribute, String}, linkage_::Union{NamedAttribute, Attribute}, dso_local_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, thread_local__=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, value_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, alignment_=nothing::Union{Nothing, Union{NamedAttribute, Int64}}, addr_space_=nothing::Union{Nothing, Union{NamedAttribute, Int32}}, unnamed_addr_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, section_=nothing::Union{Nothing, Union{NamedAttribute, String}}, comdat_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, visibility__=nothing::Union{Nothing, Union{NamedAttribute, Attribute}})
  results = []
  operands = []
  regions = [initializer_]
  successors = []
  attributes = [make_named_attribute("global_type", global_type_), make_named_attribute("sym_name", sym_name_), make_named_attribute("linkage", linkage_)]

  (constant_ != nothing) && push!(attributes, make_named_attribute("constant", constant_))
  (dso_local_ != nothing) && push!(attributes, make_named_attribute("dso_local", dso_local_))
  (thread_local__ != nothing) && push!(attributes, make_named_attribute("thread_local_", thread_local__))
  (value_ != nothing) && push!(attributes, make_named_attribute("value", value_))
  (alignment_ != nothing) && push!(attributes, make_named_attribute("alignment", alignment_))
  (addr_space_ != nothing) && push!(attributes, make_named_attribute("addr_space", addr_space_))
  (unnamed_addr_ != nothing) && push!(attributes, make_named_attribute("unnamed_addr", unnamed_addr_))
  (section_ != nothing) && push!(attributes, make_named_attribute("section", section_))
  (comdat_ != nothing) && push!(attributes, make_named_attribute("comdat", comdat_))
  (visibility__ != nothing) && push!(attributes, make_named_attribute("visibility_", visibility__))

  create_operation(
        "llvm.mlir.global", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
icmp

"""
function ICmp(; location::Location, res_::MLIRType, lhs_::Value, rhs_::Value, predicate_::Union{NamedAttribute, Attribute})
  results = [res_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = [make_named_attribute("predicate", predicate_)]
  
  create_operation(
        "llvm.icmp", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
inline_asm

The InlineAsmOp mirrors the underlying LLVM semantics with a notable
exception: the embedded `asm_string` is not allowed to define or reference
any symbol or any global variable: only the operands of the op may be read,
written, or referenced.
Attempting to define or reference any symbol or any global behavior is
considered undefined behavior at this time.
  
"""
function InlineAsm(; location::Location, res_=nothing::Union{Nothing, MLIRType}, operands_::Vector{Value}, asm_string_::Union{NamedAttribute, String}, constraints_::Union{NamedAttribute, String}, has_side_effects_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, is_align_stack_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, asm_dialect_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, operand_attrs_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}})
  results = []
  (res_ != nothing) && push!(results, res_)
  operands = [operands_...]
  regions = []
  successors = []
  attributes = [make_named_attribute("asm_string", asm_string_), make_named_attribute("constraints", constraints_)]

  (has_side_effects_ != nothing) && push!(attributes, make_named_attribute("has_side_effects", has_side_effects_))
  (is_align_stack_ != nothing) && push!(attributes, make_named_attribute("is_align_stack", is_align_stack_))
  (asm_dialect_ != nothing) && push!(attributes, make_named_attribute("asm_dialect", asm_dialect_))
  (operand_attrs_ != nothing) && push!(attributes, make_named_attribute("operand_attrs", operand_attrs_))

  create_operation(
        "llvm.inline_asm", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
insertelement

"""
function InsertElement(; location::Location, res_::MLIRType, vector_::Value, value_::Value, position_::Value)
  results = [res_]
  operands = [vector_, value_, position_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.insertelement", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
insertvalue

"""
function InsertValue(; location::Location, res_::MLIRType, container_::Value, value_::Value, position_::Union{NamedAttribute, Attribute})
  results = [res_]
  operands = [container_, value_]
  regions = []
  successors = []
  attributes = [make_named_attribute("position", position_)]
  
  create_operation(
        "llvm.insertvalue", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
inttoptr

"""
function IntToPtr(; location::Location, res_::MLIRType, arg_::Value)
  results = [res_]
  operands = [arg_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.inttoptr", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
invoke

"""
function Invoke(; location::Location, _unnamed0_::Vector{MLIRType}, callee_operands_::Vector{Value}, normalDestOperands_::Vector{Value}, unwindDestOperands_::Vector{Value}, normalDest_::Block, unwindDest_::Block, callee_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, branch_weights_=nothing::Union{Nothing, Union{NamedAttribute, Vector{Float32}}})
  results = [_unnamed0_...]
  operands = [callee_operands_..., normalDestOperands_..., unwindDestOperands_...]
  regions = []
  successors = [normalDest_, unwindDest_]
  attributes = []

  (callee_ != nothing) && push!(attributes, make_named_attribute("callee", callee_))
  (branch_weights_ != nothing) && push!(attributes, make_named_attribute("branch_weights", branch_weights_))

  push!(attributes, make_named_attribute("operand_segment_sizes", Int32[length(callee_operands_), length(normalDestOperands_), length(unwindDestOperands_)]))
  
  create_operation(
        "llvm.invoke", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
func

MLIR functions are defined by an operation that is not built into the IR
itself. The LLVM dialect provides an `llvm.func` operation to define
functions compatible with LLVM IR. These functions have LLVM dialect
function type but use MLIR syntax to express it. They are required to have
exactly one result type. LLVM function operation is intended to capture
additional properties of LLVM functions, such as linkage and calling
convention, that may be modeled differently by the built-in MLIR function.

```mlir
// The type of @bar is !llvm<\"i64 (i64)\">
llvm.func @bar(%arg0: i64) -> i64 {
  llvm.return %arg0 : i64
}

// Type type of @foo is !llvm<\"void (i64)\">
// !llvm.void type is omitted
llvm.func @foo(%arg0: i64) {
  llvm.return
}

// A function with `internal` linkage.
llvm.func internal @internal_func() {
  llvm.return
}
```
  
"""
function LLVMFunc(; location::Location, body_::Region, sym_name_::Union{NamedAttribute, String}, function_type_::Union{NamedAttribute, Attribute}, linkage_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, dso_local_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, CConv_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, comdat_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, personality_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, garbageCollector_=nothing::Union{Nothing, Union{NamedAttribute, String}}, passthrough_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, arg_attrs_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, res_attrs_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, function_entry_count_=nothing::Union{Nothing, Union{NamedAttribute, Int64}}, memory_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, visibility__=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, arm_streaming_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, arm_locally_streaming_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, section_=nothing::Union{Nothing, Union{NamedAttribute, String}}, unnamed_addr_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, alignment_=nothing::Union{Nothing, Union{NamedAttribute, Int64}})
  results = []
  operands = []
  regions = [body_]
  successors = []
  attributes = [make_named_attribute("sym_name", sym_name_), make_named_attribute("function_type", function_type_)]

  (linkage_ != nothing) && push!(attributes, make_named_attribute("linkage", linkage_))
  (dso_local_ != nothing) && push!(attributes, make_named_attribute("dso_local", dso_local_))
  (CConv_ != nothing) && push!(attributes, make_named_attribute("CConv", CConv_))
  (comdat_ != nothing) && push!(attributes, make_named_attribute("comdat", comdat_))
  (personality_ != nothing) && push!(attributes, make_named_attribute("personality", personality_))
  (garbageCollector_ != nothing) && push!(attributes, make_named_attribute("garbageCollector", garbageCollector_))
  (passthrough_ != nothing) && push!(attributes, make_named_attribute("passthrough", passthrough_))
  (arg_attrs_ != nothing) && push!(attributes, make_named_attribute("arg_attrs", arg_attrs_))
  (res_attrs_ != nothing) && push!(attributes, make_named_attribute("res_attrs", res_attrs_))
  (function_entry_count_ != nothing) && push!(attributes, make_named_attribute("function_entry_count", function_entry_count_))
  (memory_ != nothing) && push!(attributes, make_named_attribute("memory", memory_))
  (visibility__ != nothing) && push!(attributes, make_named_attribute("visibility_", visibility__))
  (arm_streaming_ != nothing) && push!(attributes, make_named_attribute("arm_streaming", arm_streaming_))
  (arm_locally_streaming_ != nothing) && push!(attributes, make_named_attribute("arm_locally_streaming", arm_locally_streaming_))
  (section_ != nothing) && push!(attributes, make_named_attribute("section", section_))
  (unnamed_addr_ != nothing) && push!(attributes, make_named_attribute("unnamed_addr", unnamed_addr_))
  (alignment_ != nothing) && push!(attributes, make_named_attribute("alignment", alignment_))

  create_operation(
        "llvm.func", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
lshr

"""
function LShr(; location::Location, res_::MLIRType, lhs_::Value, rhs_::Value)
  results = [res_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.lshr", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
landingpad

"""
function Landingpad(; location::Location, res_::MLIRType, _unnamed0_::Vector{Value}, cleanup_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}})
  results = [res_]
  operands = [_unnamed0_...]
  regions = []
  successors = []
  attributes = []

  (cleanup_ != nothing) && push!(attributes, make_named_attribute("cleanup", cleanup_))

  create_operation(
        "llvm.landingpad", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
load

The `load` operation is used to read from memory. A load may be marked as
atomic, volatile, and/or nontemporal, and takes a number of optional
attributes that specify aliasing information.

An atomic load only supports a limited set of pointer, integer, and
floating point types, and requires an explicit alignment.

Examples:
```mlir
// A volatile load of a float variable.
%0 = llvm.load volatile %ptr : !llvm.ptr -> f32

// A nontemporal load of a float variable.
%0 = llvm.load %ptr {nontemporal} : !llvm.ptr -> f32

// An atomic load of an integer variable.
%0 = llvm.load %ptr atomic monotonic {alignment = 8 : i64}
    : !llvm.ptr -> i64
```

See the following link for more details:
https://llvm.org/docs/LangRef.html#load-instruction
  
"""
function Load(; location::Location, res_::MLIRType, addr_::Value, alignment_=nothing::Union{Nothing, Union{NamedAttribute, Int64}}, volatile__=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, nontemporal_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, ordering_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, syncscope_=nothing::Union{Nothing, Union{NamedAttribute, String}}, access_groups_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, alias_scopes_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, noalias_scopes_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, tbaa_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}})
  results = [res_]
  operands = [addr_]
  regions = []
  successors = []
  attributes = []

  (alignment_ != nothing) && push!(attributes, make_named_attribute("alignment", alignment_))
  (volatile__ != nothing) && push!(attributes, make_named_attribute("volatile_", volatile__))
  (nontemporal_ != nothing) && push!(attributes, make_named_attribute("nontemporal", nontemporal_))
  (ordering_ != nothing) && push!(attributes, make_named_attribute("ordering", ordering_))
  (syncscope_ != nothing) && push!(attributes, make_named_attribute("syncscope", syncscope_))
  (access_groups_ != nothing) && push!(attributes, make_named_attribute("access_groups", access_groups_))
  (alias_scopes_ != nothing) && push!(attributes, make_named_attribute("alias_scopes", alias_scopes_))
  (noalias_scopes_ != nothing) && push!(attributes, make_named_attribute("noalias_scopes", noalias_scopes_))
  (tbaa_ != nothing) && push!(attributes, make_named_attribute("tbaa", tbaa_))

  create_operation(
        "llvm.load", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
mul

"""
function Mul(; location::Location, res_::MLIRType, lhs_::Value, rhs_::Value)
  results = [res_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.mul", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
mlir.null

Unlike LLVM IR, MLIR does not have first-class null pointers. They must be
explicitly created as SSA values using `llvm.mlir.null`. This operation has
no operands or attributes, and returns a null value of a wrapped LLVM IR
pointer type.

Examples:

```mlir
// Null pointer to i8.
%0 = llvm.mlir.null : !llvm.ptr<i8>

// Null pointer to a function with signature void().
%1 = llvm.mlir.null : !llvm.ptr<func<void ()>>
```
  
"""
function Null(; location::Location, res_::MLIRType)
  results = [res_]
  operands = []
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.mlir.null", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
or

"""
function Or(; location::Location, res_::MLIRType, lhs_::Value, rhs_::Value)
  results = [res_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.or", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
mlir.poison

Unlike LLVM IR, MLIR does not have first-class poison values. Such values
must be created as SSA values using `llvm.mlir.poison`. This operation has
no operands or attributes. It creates a poison value of the specified LLVM
IR dialect type.

Example:

```mlir
// Create a poison value for a structure with a 32-bit integer followed
// by a float.
%0 = llvm.mlir.poison : !llvm.struct<(i32, f32)>
```
  
"""
function Poison(; location::Location, res_::MLIRType)
  results = [res_]
  operands = []
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.mlir.poison", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
ptrtoint

"""
function PtrToInt(; location::Location, res_::MLIRType, arg_::Value)
  results = [res_]
  operands = [arg_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.ptrtoint", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
resume

"""
function Resume(; location::Location, value_::Value)
  results = []
  operands = [value_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.resume", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
return

"""
function Return(; location::Location, arg_=nothing::Union{Nothing, Value})
  results = []
  operands = []

  (arg_ != nothing) && push!(operands, arg_)
regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.return", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
sdiv

"""
function SDiv(; location::Location, res_::MLIRType, lhs_::Value, rhs_::Value)
  results = [res_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.sdiv", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
sext

"""
function SExt(; location::Location, res_::MLIRType, arg_::Value)
  results = [res_]
  operands = [arg_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.sext", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
sitofp

"""
function SIToFP(; location::Location, res_::MLIRType, arg_::Value)
  results = [res_]
  operands = [arg_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.sitofp", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
srem

"""
function SRem(; location::Location, res_::MLIRType, lhs_::Value, rhs_::Value)
  results = [res_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.srem", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
select

"""
function Select(; location::Location, res_::MLIRType, condition_::Value, trueValue_::Value, falseValue_::Value, fastmathFlags_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}})
  results = [res_]
  operands = [condition_, trueValue_, falseValue_]
  regions = []
  successors = []
  attributes = []

  (fastmathFlags_ != nothing) && push!(attributes, make_named_attribute("fastmathFlags", fastmathFlags_))

  create_operation(
        "llvm.select", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
shl

"""
function Shl(; location::Location, res_::MLIRType, lhs_::Value, rhs_::Value)
  results = [res_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.shl", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
shufflevector

"""
function ShuffleVector(; location::Location, res_::MLIRType, v1_::Value, v2_::Value, mask_::Union{NamedAttribute, Vector{Float32}})
  results = [res_]
  operands = [v1_, v2_]
  regions = []
  successors = []
  attributes = [make_named_attribute("mask", mask_)]
  
  create_operation(
        "llvm.shufflevector", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
store

The `store` operation is used to write to memory. A store may be marked as
atomic, volatile, and/or nontemporal, and takes a number of optional
attributes that specify aliasing information.

An atomic store only supports a limited set of pointer, integer, and
floating point types, and requires an explicit alignment.

Examples:
```mlir
// A volatile store of a float variable.
llvm.store volatile %val, %ptr : f32, !llvm.ptr

// A nontemporal store of a float variable.
llvm.store %val, %ptr {nontemporal} : f32, !llvm.ptr

// An atomic store of an integer variable.
llvm.store %val, %ptr atomic monotonic {alignment = 8 : i64}
    : i64, !llvm.ptr
```

See the following link for more details:
https://llvm.org/docs/LangRef.html#store-instruction
  
"""
function Store(; location::Location, value_::Value, addr_::Value, alignment_=nothing::Union{Nothing, Union{NamedAttribute, Int64}}, volatile__=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, nontemporal_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, ordering_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, syncscope_=nothing::Union{Nothing, Union{NamedAttribute, String}}, access_groups_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, alias_scopes_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, noalias_scopes_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, tbaa_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}})
  results = []
  operands = [value_, addr_]
  regions = []
  successors = []
  attributes = []

  (alignment_ != nothing) && push!(attributes, make_named_attribute("alignment", alignment_))
  (volatile__ != nothing) && push!(attributes, make_named_attribute("volatile_", volatile__))
  (nontemporal_ != nothing) && push!(attributes, make_named_attribute("nontemporal", nontemporal_))
  (ordering_ != nothing) && push!(attributes, make_named_attribute("ordering", ordering_))
  (syncscope_ != nothing) && push!(attributes, make_named_attribute("syncscope", syncscope_))
  (access_groups_ != nothing) && push!(attributes, make_named_attribute("access_groups", access_groups_))
  (alias_scopes_ != nothing) && push!(attributes, make_named_attribute("alias_scopes", alias_scopes_))
  (noalias_scopes_ != nothing) && push!(attributes, make_named_attribute("noalias_scopes", noalias_scopes_))
  (tbaa_ != nothing) && push!(attributes, make_named_attribute("tbaa", tbaa_))

  create_operation(
        "llvm.store", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
sub

"""
function Sub(; location::Location, res_::MLIRType, lhs_::Value, rhs_::Value)
  results = [res_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.sub", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
switch

"""
function Switch(; location::Location, value_::Value, defaultOperands_::Vector{Value}, caseOperands_::Vector{Value}, defaultDestination_::Block, caseDestinations_::Vector{Block}, case_values_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, case_operand_segments_::Union{NamedAttribute, Vector{Float32}}, branch_weights_=nothing::Union{Nothing, Union{NamedAttribute, Vector{Float32}}})
  results = []
  operands = [value_, defaultOperands_..., caseOperands_...]
  regions = []
  successors = [defaultDestination_, caseDestinations_...]
  attributes = [make_named_attribute("case_operand_segments", case_operand_segments_)]

  (case_values_ != nothing) && push!(attributes, make_named_attribute("case_values", case_values_))
  (branch_weights_ != nothing) && push!(attributes, make_named_attribute("branch_weights", branch_weights_))

  push!(attributes, make_named_attribute("operand_segment_sizes", Int32[1, length(defaultOperands_), length(caseOperands_)]))
  
  create_operation(
        "llvm.switch", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
trunc

"""
function Trunc(; location::Location, res_::MLIRType, arg_::Value)
  results = [res_]
  operands = [arg_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.trunc", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
udiv

"""
function UDiv(; location::Location, res_::MLIRType, lhs_::Value, rhs_::Value)
  results = [res_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.udiv", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
uitofp

"""
function UIToFP(; location::Location, res_::MLIRType, arg_::Value)
  results = [res_]
  operands = [arg_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.uitofp", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
urem

"""
function URem(; location::Location, res_::MLIRType, lhs_::Value, rhs_::Value)
  results = [res_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.urem", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
mlir.undef

Unlike LLVM IR, MLIR does not have first-class undefined values. Such values
must be created as SSA values using `llvm.mlir.undef`. This operation has no
operands or attributes. It creates an undefined value of the specified LLVM
IR dialect type.

Example:

```mlir
// Create a structure with a 32-bit integer followed by a float.
%0 = llvm.mlir.undef : !llvm.struct<(i32, f32)>
```
  
"""
function Undef(; location::Location, res_::MLIRType)
  results = [res_]
  operands = []
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.mlir.undef", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
unreachable

"""
function Unreachable(; location::Location)
  results = []
  operands = []
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.unreachable", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
xor

"""
function XOr(; location::Location, res_::MLIRType, lhs_::Value, rhs_::Value)
  results = [res_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.xor", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
zext

"""
function ZExt(; location::Location, res_::MLIRType, arg_::Value)
  results = [res_]
  operands = [arg_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "llvm.zext", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


end #Llvm
