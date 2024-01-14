module cf

import ...IR: NamedAttribute, MLIRType, Location, Block, Region, Attribute, create_operation, context, IndexType, get_value
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`assert`

Assert operation at runtime with single boolean operand and an error
message attribute.
If the argument is `true` this operation has no effect. Otherwise, the
program execution will abort. The provided error message may be used by a
runtime to propagate the error to the user.

# Example

```mlir
assert %b, \"Expected ... to be true\"
```
"""
function assert(arg; msg, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value(arg), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("msg", msg), ]
    
    create_operation(
        "cf.assert", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`br`

The `cf.br` operation represents a direct branch operation to a given
block. The operands of this operation are forwarded to the successor block,
and the number and type of the operands must match the arguments of the
target block.

# Example

```mlir
^bb2:
  %2 = call @someFn()
  cf.br ^bb3(%2 : tensor<*xf32>)
^bb3(%3: tensor<*xf32>):
```
"""
function br(destOperands; dest::Block, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value.(destOperands)..., ]
    owned_regions = Region[]
    successors = Block[dest, ]
    attributes = NamedAttribute[]
    
    create_operation(
        "cf.br", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`cond_br`

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

# Example

```mlir
func.func @select(%a: i32, %b: i32, %flag: i1) -> i32 {
  // Both targets are the same, operands differ
  cond_br %flag, ^bb1(%a : i32), ^bb1(%b : i32)

^bb1(%x : i32) :
  return %x : i32
}
```
"""
function cond_br(condition, trueDestOperands, falseDestOperands; trueDest::Block, falseDest::Block, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value(condition), get_value.(trueDestOperands)..., get_value.(falseDestOperands)..., ]
    owned_regions = Region[]
    successors = Block[trueDest, falseDest, ]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([1, length(trueDestOperands), length(falseDestOperands), ]))
    
    create_operation(
        "cf.cond_br", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`switch`

The `switch` terminator operation represents a switch on a signless integer
value. If the flag matches one of the specified cases, then the
corresponding destination is jumped to. If the flag does not match any of
the cases, the default destination is jumped to. The count and types of
operands must align with the arguments in the corresponding target blocks.

# Example

```mlir
switch %flag : i32, [
  default: ^bb1(%a : i32),
  42: ^bb1(%b : i32),
  43: ^bb3(%c : i32)
]
```
"""
function switch(flag, defaultOperands, caseOperands; case_values=nothing, case_operand_segments, defaultDestination::Block, caseDestinations::Vector{Block}, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value(flag), get_value.(defaultOperands)..., get_value.(caseOperands)..., ]
    owned_regions = Region[]
    successors = Block[defaultDestination, caseDestinations..., ]
    attributes = NamedAttribute[namedattribute("case_operand_segments", case_operand_segments), ]
    push!(attributes, operandsegmentsizes([1, length(defaultOperands), length(caseOperands), ]))
    (case_values != nothing) && push!(attributes, namedattribute("case_values", case_values))
    
    create_operation(
        "cf.switch", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # cf
