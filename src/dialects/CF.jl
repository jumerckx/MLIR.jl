module Cf

import ...IR: context, NamedAttribute, MLIRType, Value, Location, Block, Attribute, ArrayAttribute, create_operation
import ...API

make_named_attribute(name, val) = make_named_attribute(name, Attribute(val))

make_named_attribute(name, val::Attribute) = NamedAttribute(name, val)

function make_named_attribute(name, val::NamedAttribute)
  assert(true) # TODO(jm): check whether name of attribute is correct, getting the name might need to be added to IR.jl?
  return val
end

"""
assert

Assert operation with single boolean operand and an error message attribute.
If the argument is `true` this operation has no effect. Otherwise, the
program execution will abort. The provided error message may be used by a
runtime to propagate the error to the user.

Example:

```mlir
assert %b, \"Expected ... to be true\"
```
  
"""
function Assert(; location::Location, arg_::Value, msg_::Union{NamedAttribute, String})
  results = []
  operands = [arg_]
  successors = []
  attributes = [make_named_attribute("msg", msg_)]
  
  create_operation(
        "cf.assert", location, 
        results = results, 
        operands = operands,
        owned_regions = [], 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
br

The `cf.br` operation represents a direct branch operation to a given
block. The operands of this operation are forwarded to the successor block,
and the number and type of the operands must match the arguments of the
target block.

Example:

```mlir
^bb2:
  %2 = call @someFn()
  cf.br ^bb3(%2 : tensor<*xf32>)
^bb3(%3: tensor<*xf32>):
```
  
"""
function Branch(; location::Location, destOperands_::Vector{Value}, dest_::Block)
  results = []
  operands = [destOperands_...]
  successors = [dest_]
  attributes = []
  
  create_operation(
        "cf.br", location, 
        results = results, 
        operands = operands,
        owned_regions = [], 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
cond_br

The `cond_br` terminator operation represents a conditional branch on a
boolean (1-bit integer) value. If the bit is set, then the first destination
is jumped to; if it is false, the second destination is chosen. The count
and types of operands must align with the arguments in the corresponding
target blocks.

The MLIR conditional branch operation is not allowed to target the entry
block for a region. The two destinations of the conditional branch operation
are allowed to be the same.

The following example illustrates a function with a conditional branch
operation that targets the same block.

Example:

```mlir
func.func @select(%a: i32, %b: i32, %flag: i1) -> i32 {
  // Both targets are the same, operands differ
  cond_br %flag, ^bb1(%a : i32), ^bb1(%b : i32)

^bb1(%x : i32) :
  return %x : i32
}
```
  
"""
function CondBranch(; location::Location, condition_::Value, trueDestOperands_::Vector{Value}, falseDestOperands_::Vector{Value}, trueDest_::Block, falseDest_::Block)
  results = MLIRType[]
  operands = Value[condition_, trueDestOperands_..., falseDestOperands_...]
  successors = Block[trueDest_, falseDest_]
  attributes = NamedAttribute[]

  # push!(attributes, NamedAttribute("operand_segment_sizes", Attribute(Int32[1, length(trueDestOperands_), length(falseDestOperands_)])))
  push!(attributes, make_named_attribute("operand_segment_sizes", Attribute(API.mlirDenseI32ArrayGet(
    context().context,
    3,
    Int32[1, length(trueDestOperands_), length(falseDestOperands_)]))))
  
  create_operation(
        "cf.cond_br", location, 
        results = results, 
        operands = operands,
        owned_regions = [], 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
switch

The `switch` terminator operation represents a switch on a signless integer
value. If the flag matches one of the specified cases, then the
corresponding destination is jumped to. If the flag does not match any of
the cases, the default destination is jumped to. The count and types of
operands must align with the arguments in the corresponding target blocks.

Example:

```mlir
switch %flag : i32, [
  default: ^bb1(%a : i32),
  42: ^bb1(%b : i32),
  43: ^bb3(%c : i32)
]
```
  
"""
function Switch(; location::Location, flag_::Value, defaultOperands_::Vector{Value}, caseOperands_::Vector{Value}, defaultDestination_::Block, caseDestinations_::Vector{Block}, case_values_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, case_operand_segments_::Union{NamedAttribute, Vector{Float32}})
  results = []
  operands = [flag_, defaultOperands_..., caseOperands_...]
  successors = [defaultDestination_, caseDestinations_...]
  attributes = [make_named_attribute("case_operand_segments", case_operand_segments_)]

  (case_values_ != nothing) && push!(attributes, make_named_attribute("case_values", case_values_))

  push!(attributes, make_named_attribute("operand_segment_sizes", Int32[1, length(defaultOperands_), length(caseOperands_)]))
  
  create_operation(
        "cf.switch", location, 
        results = results, 
        operands = operands,
        owned_regions = [], 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


end #Cf
