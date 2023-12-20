module Index

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation
import ..Dialects: make_named_attribute

"""
add

The `index.add` operation takes two index values and computes their sum.

Example:

```mlir
// c = a + b
%c = index.add %a, %b
```
  
"""
function Add(; location::Location, result_::MLIRType, lhs_::Value, rhs_::Value)
  results = [result_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "index.add", location, 
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

The `index.and` operation takes two index values and computes their bitwise
and.

Example:

```mlir
// c = a & b
%c = index.and %a, %b
```
  
"""
function And(; location::Location, result_::MLIRType, lhs_::Value, rhs_::Value)
  results = [result_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "index.and", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
bool.constant

The `index.bool.constant` operation produces an bool-typed SSA value equal
to either `true` or `false`.

This operation is used to materialize bool constants that arise when folding
`index.cmp`.

Example:

```mlir
%0 = index.bool.constant true
```
  
"""
function BoolConstant(; location::Location, result_::MLIRType, value_::Union{NamedAttribute, Bool})
  results = [result_]
  operands = []
  regions = []
  successors = []
  attributes = [make_named_attribute("value", value_)]
  
  create_operation(
        "index.bool.constant", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
casts

The `index.casts` operation enables conversions between values of index type
and concrete fixed-width integer types. If casting to a wider integer, the
value is sign-extended. If casting to a narrower integer, the value is
truncated.

Example:

```mlir
// Cast to i32
%0 = index.casts %a : index to i32

// Cast from i64
%1 = index.casts %b : i64 to index
```
  
"""
function CastS(; location::Location, output_::MLIRType, input_::Value)
  results = [output_]
  operands = [input_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "index.casts", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
castu

The `index.castu` operation enables conversions between values of index type
and concrete fixed-width integer types. If casting to a wider integer, the
value is zero-extended. If casting to a narrower integer, the value is
truncated.

Example:

```mlir
// Cast to i32
%0 = index.castu %a : index to i32

// Cast from i64
%1 = index.castu %b : i64 to index
```
  
"""
function CastU(; location::Location, output_::MLIRType, input_::Value)
  results = [output_]
  operands = [input_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "index.castu", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
ceildivs

The `index.ceildivs` operation takes two index values and computes their
signed quotient. Treats the leading bit as the sign and rounds towards
positive infinity, i.e. `7 / -2 = -3`.

Note: division by zero and signed division overflow are undefined behaviour.

Example:

```mlir
// c = ceil(a / b)
%c = index.ceildivs %a, %b
```
  
"""
function CeilDivS(; location::Location, result_::MLIRType, lhs_::Value, rhs_::Value)
  results = [result_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "index.ceildivs", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
ceildivu

The `index.ceildivu` operation takes two index values and computes their
unsigned quotient. Treats the leading bit as the most significant and rounds
towards positive infinity, i.e. `6 / -2 = 1`.

Note: division by zero is undefined behaviour.

Example:

```mlir
// c = ceil(a / b)
%c = index.ceildivu %a, %b
```
  
"""
function CeilDivU(; location::Location, result_::MLIRType, lhs_::Value, rhs_::Value)
  results = [result_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "index.ceildivu", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
cmp

The `index.cmp` operation takes two index values and compares them according
to the comparison predicate and returns an `i1`. The following comparisons
are supported:

-   `eq`:  equal
-   `ne`:  not equal
-   `slt`: signed less than
-   `sle`: signed less than or equal
-   `sgt`: signed greater than
-   `sge`: signed greater than or equal
-   `ult`: unsigned less than
-   `ule`: unsigned less than or equal
-   `ugt`: unsigned greater than
-   `uge`: unsigned greater than or equal

The result is `1` if the comparison is true and `0` otherwise.

Example:

```mlir
// Signed less than comparison.
%0 = index.cmp slt(%a, %b)

// Unsigned greater than or equal comparison.
%1 = index.cmp uge(%a, %b)

// Not equal comparison.
%2 = index.cmp ne(%a, %b)
```
  
"""
function Cmp(; location::Location, result_::MLIRType, lhs_::Value, rhs_::Value, pred_::Union{NamedAttribute, Attribute})
  results = [result_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = [make_named_attribute("pred", pred_)]
  
  create_operation(
        "index.cmp", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
constant

The `index.constant` operation produces an index-typed SSA value equal to
some index-typed integer constant.

Example:

```mlir
%0 = index.constant 42
```
  
"""
function Constant(; location::Location, result_::MLIRType, value_::Union{NamedAttribute, Attribute})
  results = [result_]
  operands = []
  regions = []
  successors = []
  attributes = [make_named_attribute("value", value_)]
  
  create_operation(
        "index.constant", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
divs

The `index.divs` operation takes two index values and computes their signed
quotient. Treats the leading bit as the sign and rounds towards zero, i.e.
`6 / -2 = -3`.

Note: division by zero and signed division overflow are undefined behaviour.

Example:

```mlir
// c = a / b
%c = index.divs %a, %b
```
  
"""
function DivS(; location::Location, result_::MLIRType, lhs_::Value, rhs_::Value)
  results = [result_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "index.divs", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
divu

The `index.divu` operation takes two index values and computes their
unsigned quotient. Treats the leading bit as the most significant and rounds
towards zero, i.e. `6 / -2 = 0`.

Note: division by zero is undefined behaviour.

Example:

```mlir
// c = a / b
%c = index.divu %a, %b
```
  
"""
function DivU(; location::Location, result_::MLIRType, lhs_::Value, rhs_::Value)
  results = [result_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "index.divu", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
floordivs

The `index.floordivs` operation takes two index values and computes their
signed quotient. Treats the leading bit as the sign and rounds towards
negative infinity, i.e. `5 / -2 = -3`.

Note: division by zero and signed division overflow are undefined behaviour.

Example:

```mlir
// c = floor(a / b)
%c = index.floordivs %a, %b
```
  
"""
function FloorDivS(; location::Location, result_::MLIRType, lhs_::Value, rhs_::Value)
  results = [result_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "index.floordivs", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
maxs

The `index.maxs` operation takes two index values and computes their signed
maximum value. Treats the leading bit as the sign, i.e. `max(-2, 6) = 6`.

Example:

```mlir
// c = max(a, b)
%c = index.maxs %a, %b
```
  
"""
function MaxS(; location::Location, result_::MLIRType, lhs_::Value, rhs_::Value)
  results = [result_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "index.maxs", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
maxu

The `index.maxu` operation takes two index values and computes their
unsigned maximum value. Treats the leading bit as the most significant, i.e.
`max(15, 6) = 15` or `max(-2, 6) = -2`.

Example:

```mlir
// c = max(a, b)
%c = index.maxu %a, %b
```
  
"""
function MaxU(; location::Location, result_::MLIRType, lhs_::Value, rhs_::Value)
  results = [result_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "index.maxu", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
mins

The `index.mins` operation takes two index values and computes their signed
minimum value. Treats the leading bit as the sign, i.e. `min(-2, 6) = -2`.

Example:

```mlir
// c = min(a, b)
%c = index.mins %a, %b
```
  
"""
function MinS(; location::Location, result_::MLIRType, lhs_::Value, rhs_::Value)
  results = [result_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "index.mins", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
minu

The `index.minu` operation takes two index values and computes their
unsigned minimum value. Treats the leading bit as the most significant, i.e.
`min(15, 6) = 6` or `min(-2, 6) = 6`.

Example:

```mlir
// c = min(a, b)
%c = index.minu %a, %b
```
  
"""
function MinU(; location::Location, result_::MLIRType, lhs_::Value, rhs_::Value)
  results = [result_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "index.minu", location, 
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

The `index.mul` operation takes two index values and computes their product.

Example:

```mlir
// c = a * b
%c = index.mul %a, %b
```
  
"""
function Mul(; location::Location, result_::MLIRType, lhs_::Value, rhs_::Value)
  results = [result_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "index.mul", location, 
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

The `index.or` operation takes two index values and computes their bitwise
or.

Example:

```mlir
// c = a | b
%c = index.or %a, %b
```
  
"""
function Or(; location::Location, result_::MLIRType, lhs_::Value, rhs_::Value)
  results = [result_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "index.or", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
rems

The `index.rems` operation takes two index values and computes their signed
remainder. Treats the leading bit as the sign, i.e. `6 % -2 = 0`.

Example:

```mlir
// c = a % b
%c = index.rems %a, %b
```
  
"""
function RemS(; location::Location, result_::MLIRType, lhs_::Value, rhs_::Value)
  results = [result_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "index.rems", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
remu

The `index.remu` operation takes two index values and computes their
unsigned remainder. Treats the leading bit as the most significant, i.e.
`6 % -2 = 6`.

Example:

```mlir
// c = a % b
%c = index.remu %a, %b
```
  
"""
function RemU(; location::Location, result_::MLIRType, lhs_::Value, rhs_::Value)
  results = [result_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "index.remu", location, 
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

The `index.shl` operation shifts an index value to the left by a variable
amount. The low order bits are filled with zeroes. The RHS operand is always
treated as unsigned. If the RHS operand is equal to or greater than the
index bitwidth, the operation is undefined.

Example:

```mlir
// c = a << b
%c = index.shl %a, %b
```
  
"""
function Shl(; location::Location, result_::MLIRType, lhs_::Value, rhs_::Value)
  results = [result_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "index.shl", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
shrs

The `index.shrs` operation shifts an index value to the right by a variable
amount. The LHS operand is treated as signed. The high order bits are filled
with copies of the most significant bit. If the RHS operand is equal to or
greater than the index bitwidth, the operation is undefined.

Example:

```mlir
// c = a >> b
%c = index.shrs %a, %b
```
  
"""
function ShrS(; location::Location, result_::MLIRType, lhs_::Value, rhs_::Value)
  results = [result_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "index.shrs", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
shru

The `index.shru` operation shifts an index value to the right by a variable
amount. The LHS operand is treated as unsigned. The high order bits are
filled with zeroes. If the RHS operand is equal to or greater than the index
bitwidth, the operation is undefined.

Example:

```mlir
// c = a >> b
%c = index.shru %a, %b
```
  
"""
function ShrU(; location::Location, result_::MLIRType, lhs_::Value, rhs_::Value)
  results = [result_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "index.shru", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
sizeof

The `index.sizeof` operation produces an index-typed SSA value equal to the
size in bits of the `index` type. For example, on 32-bit systems, the result
is `32 : index`, and on 64-bit systems, the result is `64 : index`.

Example:

```mlir
%0 = index.sizeof
```
  
"""
function SizeOf(; location::Location, result_::MLIRType)
  results = [result_]
  operands = []
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "index.sizeof", location, 
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

The `index.sub` operation takes two index values and computes the difference
of the first from the second operand.

Example:

```mlir
// c = a - b
%c = index.sub %a, %b
```
  
"""
function Sub(; location::Location, result_::MLIRType, lhs_::Value, rhs_::Value)
  results = [result_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "index.sub", location, 
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

The `index.xor` operation takes two index values and computes their bitwise
xor.

Example:

```mlir
// c = a ^ b
%c = index.xor %a, %b
```
  
"""
function XOr(; location::Location, result_::MLIRType, lhs_::Value, rhs_::Value)
  results = [result_]
  operands = [lhs_, rhs_]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "index.xor", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


end #Index
