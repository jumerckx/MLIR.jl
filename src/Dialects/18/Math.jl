module math

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`absf`

The `absf` operation computes the absolute value. It takes one operand of
floating point type (i.e., scalar, tensor or vector) and returns one result
of the same type.

# Example

```mlir
// Scalar absolute value.
%a = math.absf %b : f64
```
"""
function absf(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.absf", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`absi`

The `absi` operation computes the absolute value. It takes one operand of
integer type (i.e., scalar, tensor or vector) and returns one result of the
same type.

# Example

```mlir
// Scalar absolute value.
%a = math.absi %b : i64
```
"""
function absi(operand::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    create_operation(
        "math.absi", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`acos`

The `acos` operation computes the arcus cosine of a given value. It takes one
operand of floating point type (i.e., scalar, tensor or vector) and returns one
result of the same type.  It has no standard attributes.

# Example

```mlir
// Scalar arcus cosine value.
%a = math.acos %b : f64
```
"""
function acos(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.acos", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`acosh`

# Syntax

```
operation ::= ssa-id `=` `math.acosh` ssa-use `:` type
```

The `acosh` operation computes the arcus cosine of a given value.  It takes
one operand of floating point type (i.e., scalar, tensor or vector) and returns
one result of the same type. It has no standard attributes.

# Example

```mlir
// Hyperbolic arcus cosine of scalar value.
%a = math.acosh %b : f64
```
"""
function acosh(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.acosh", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`asin`

# Syntax

```
operation ::= ssa-id `=` `math.asin` ssa-use `:` type
```

The `asin` operation computes the arcus sine of a given value.  It takes
one operand of floating point type (i.e., scalar, tensor or vector) and returns
one result of the same type. It has no standard attributes.

# Example

```mlir
// Arcus sine of scalar value.
%a = math.asin %b : f64
```
"""
function asin(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.asin", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`asinh`

# Syntax

```
operation ::= ssa-id `=` `math.asinh` ssa-use `:` type
```

The `asinh` operation computes the hyperbolic arcus sine of a given value.  It takes
one operand of floating point type (i.e., scalar, tensor or vector) and returns
one result of the same type. It has no standard attributes.

# Example

```mlir
// Hyperbolic arcus sine of scalar value.
%a = math.asinh %b : f64
```
"""
function asinh(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.asinh", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`atan2`

The `atan2` operation takes two operands and returns one result, all of
which must be of the same type.  The operands must be of floating point type
(i.e., scalar, tensor or vector).

The 2-argument arcus tangent `atan2(y, x)` returns the angle in the
Euclidian plane between the positive x-axis and the ray through the point
(x, y).  It is a generalization of the 1-argument arcus tangent which
returns the angle on the basis of the ratio y/x.

See also https://en.wikipedia.org/wiki/Atan2

# Example

```mlir
// Scalar variant.
%a = math.atan2 %b, %c : f32
```
"""
function atan2(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.atan2", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`atan`

The `atan` operation computes the arcus tangent of a given value.  It takes
one operand of floating point type (i.e., scalar, tensor or vector) and returns
one result of the same type. It has no standard attributes.

# Example

```mlir
// Arcus tangent of scalar value.
%a = math.atan %b : f64
```
"""
function atan(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.atan", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`atanh`

# Syntax

```
operation ::= ssa-id `=` `math.atanh` ssa-use `:` type
```

The `atanh` operation computes the hyperbolic arcus tangent of a given value.  It takes
one operand of floating point type (i.e., scalar, tensor or vector) and returns
one result of the same type. It has no standard attributes.

# Example

```mlir
// Hyperbolic arcus tangent of scalar value.
%a = math.atanh %b : f64
```
"""
function atanh(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.atanh", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`cbrt`

The `cbrt` operation computes the cube root. It takes one operand of
floating point type (i.e., scalar, tensor or vector) and returns one result
of the same type. It has no standard attributes.

# Example

```mlir
// Scalar cube root value.
%a = math.cbrt %b : f64
```

Note: This op is not equivalent to powf(..., 1/3.0).
"""
function cbrt(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.cbrt", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`ceil`

The `ceil` operation computes the ceiling of a given value. It takes one
operand of floating point type (i.e., scalar, tensor or vector) and returns one
result of the same type.  It has no standard attributes.

# Example

```mlir
// Scalar ceiling value.
%a = math.ceil %b : f64
```
"""
function ceil(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.ceil", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`copysign`

The `copysign` returns a value with the magnitude of the first operand and
the sign of the second operand. It takes two operands and returns one result of
the same type. The operands must be of floating point type (i.e., scalar,
tensor or vector). It has no standard attributes.

# Example

```mlir
// Scalar copysign value.
%a = math.copysign %b, %c : f64
```
"""
function copysign(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.copysign", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`cos`

The `cos` operation computes the cosine of a given value. It takes one
operand of floating point type (i.e., scalar, tensor or vector) and returns one
result of the same type.  It has no standard attributes.

# Example

```mlir
// Scalar cosine value.
%a = math.cos %b : f64
```
"""
function cos(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.cos", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`cosh`

The `cosh` operation computes the hyperbolic cosine. It takes one operand
of floating point type (i.e., scalar, tensor or vector) and returns one
result of the same type. It has no standard attributes.

# Example

```mlir
// Scalar hyperbolic cosine value.
%a = math.cosh %b : f64
```
"""
function cosh(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.cosh", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`ctlz`

The `ctlz` operation computes the number of leading zeros of an integer value.
It operates on scalar, tensor or vector.

# Example

```mlir
// Scalar ctlz function value.
%a = math.ctlz %b : i32
```
"""
function ctlz(operand::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    create_operation(
        "math.ctlz", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`cttz`

The `cttz` operation computes the number of trailing zeros of an integer value.
It operates on scalar, tensor or vector.

# Example

```mlir
// Scalar cttz function value.
%a = math.cttz %b : i32
```
"""
function cttz(operand::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    create_operation(
        "math.cttz", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`ctpop`

The `ctpop` operation computes the number of set bits of an integer value.
It operates on scalar, tensor or vector.

# Example

```mlir
// Scalar ctpop function value.
%a = math.ctpop %b : i32
```
"""
function ctpop(operand::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    create_operation(
        "math.ctpop", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`erf`

The `erf` operation computes the error function. It takes one operand of
floating point type (i.e., scalar, tensor or vector) and returns one result of
the same type. It has no standard attributes.

# Example

```mlir
// Scalar error function value.
%a = math.erf %b : f64
```
"""
function erf(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.erf", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`exp2`

The `exp` operation takes one operand of floating point type (i.e., scalar,
tensor or vector) and returns one result of the same type. It has no standard
attributes.

# Example

```mlir
// Scalar natural exponential.
%a = math.exp2 %b : f64
```
"""
function exp2(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.exp2", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`expm1`

expm1(x) := exp(x) - 1

The `expm1` operation takes one operand of floating point type (i.e.,
scalar, tensor or vector) and returns one result of the same type. It has no
standard attributes.

# Example

```mlir
// Scalar natural exponential minus 1.
%a = math.expm1 %b : f64
```
"""
function expm1(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.expm1", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`exp`

The `exp` operation takes one operand of floating point type (i.e., scalar,
tensor or vector) and returns one result of the same type. It has no standard
attributes.

# Example

```mlir
// Scalar natural exponential.
%a = math.exp %b : f64
```
"""
function exp(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.exp", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`fpowi`

The `fpowi` operation takes a `base` operand of floating point type
(i.e. scalar, tensor or vector) and a `power` operand of integer type
(also scalar, tensor or vector) and returns one result of the same type
as `base`. The result is `base` raised to the power of `power`.
The operation is elementwise for non-scalars, e.g.:

```mlir
%v = math.fpowi %base, %power : vector<2xf32>, vector<2xi32
```

The result is a vector of:

```
[<math.fpowi %base[0], %power[0]>, <math.fpowi %base[1], %power[1]>]
```

# Example

```mlir
// Scalar exponentiation.
%a = math.fpowi %base, %power : f64, i32
```
"""
function fpowi(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.fpowi", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`floor`

The `floor` operation computes the floor of a given value. It takes one
operand of floating point type (i.e., scalar, tensor or vector) and returns one
result of the same type.  It has no standard attributes.

# Example

```mlir
// Scalar floor value.
%a = math.floor %b : f64
```
"""
function floor(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.floor", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`fma`

The `fma` operation takes three operands and returns one result, each of
these is required to be the same type. Operands must be of floating point type
(i.e., scalar, tensor or vector).

# Example

```mlir
// Scalar fused multiply-add: d = a*b + c
%d = math.fma %a, %b, %c : f64
```

The semantics of the operation correspond to those of the `llvm.fma`
[intrinsic](https://llvm.org/docs/LangRef.html#llvm-fma-intrinsic). In the
particular case of lowering to LLVM, this is guaranteed to lower
to the `llvm.fma.*` intrinsic.
"""
function fma(a::Value, b::Value, c::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[a, b, c, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.fma", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`ipowi`

The `ipowi` operation takes two operands of integer type (i.e., scalar,
tensor or vector) and returns one result of the same type. Operands
must have the same type.

# Example

```mlir
// Scalar signed integer exponentiation.
%a = math.ipowi %b, %c : i32
```
"""
function ipowi(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    create_operation(
        "math.ipowi", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`log1p`

Computes the base-e logarithm of one plus the given value. It takes one
operand of floating point type (i.e., scalar, tensor or vector) and returns one
result of the same type.

log1p(x) := log(1 + x)

# Example

```mlir
// Scalar log1p operation.
%y = math.log1p %x : f64
```
"""
function log1p(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.log1p", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`log2`

Computes the base-2 logarithm of the given value. It takes one operand of
floating point type (i.e., scalar, tensor or vector) and returns one result of
the same type.

# Example

```mlir
// Scalar log2 operation.
%y = math.log2 %x : f64
```
"""
function log2(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.log2", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`log10`

Computes the base-10 logarithm of the given value. It takes one operand of
floating point type (i.e., scalar, tensor or vector) and returns one result of
the same type.

# Example

```mlir
// Scalar log10 operation.
%y = math.log10 %x : f64
```
"""
function log10(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.log10", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`log`

Computes the base-e logarithm of the given value. It takes one operand of
floating point type (i.e., scalar, tensor or vector) and returns one result of
the same type.

# Example

```mlir
// Scalar log operation.
%y = math.log %x : f64
```
"""
function log(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.log", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`powf`

The `powf` operation takes two operands of floating point type (i.e.,
scalar, tensor or vector) and returns one result of the same type. Operands
must have the same type.

# Example

```mlir
// Scalar exponentiation.
%a = math.powf %b, %c : f64
```
"""
function powf(lhs::Value, rhs::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[lhs, rhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.powf", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`roundeven`

The `roundeven` operation returns the operand rounded to the nearest integer
value in floating-point format. It takes one operand of floating point type
(i.e., scalar, tensor or vector) and produces one result of the same type.  The
operation rounds the argument to the nearest integer value in floating-point
format, rounding halfway cases to even, regardless of the current
rounding direction.

# Example

```mlir
// Scalar round operation.
%a = math.roundeven %b : f64
```
"""
function roundeven(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.roundeven", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`round`

The `round` operation returns the operand rounded to the nearest integer
value in floating-point format. It takes one operand of floating point type
(i.e., scalar, tensor or vector) and produces one result of the same type.  The
operation rounds the argument to the nearest integer value in floating-point
format, rounding halfway cases away from zero, regardless of the current
rounding direction.

# Example

```mlir
// Scalar round operation.
%a = math.round %b : f64
```
"""
function round(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.round", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`rsqrt`

The `rsqrt` operation computes the reciprocal of the square root. It takes
one operand of floating point type (i.e., scalar, tensor or vector) and returns
one result of the same type. It has no standard attributes.

# Example

```mlir
// Scalar reciprocal square root value.
%a = math.rsqrt %b : f64
```
"""
function rsqrt(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.rsqrt", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`sin`

The `sin` operation computes the sine of a given value. It takes one
operand of floating point type (i.e., scalar, tensor or vector) and returns one
result of the same type.  It has no standard attributes.

# Example

```mlir
// Scalar sine value.
%a = math.sin %b : f64
```
"""
function sin(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.sin", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`sinh`

The `sinh` operation computes the hyperbolic sine. It takes one operand
of floating point type (i.e., scalar, tensor or vector) and returns one
result of the same type. It has no standard attributes.

# Example

```mlir
// Scalar hyperbolic sine value.
%a = math.sinh %b : f64
```
"""
function sinh(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.sinh", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`sqrt`

The `sqrt` operation computes the square root. It takes one operand of
floating point type (i.e., scalar, tensor or vector) and returns one result of
the same type. It has no standard attributes.

# Example

```mlir
// Scalar square root value.
%a = math.sqrt %b : f64
```
"""
function sqrt(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.sqrt", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`tan`

The `tan` operation computes the tangent. It takes one operand
of floating point type (i.e., scalar, tensor or vector) and returns one
result of the same type. It has no standard attributes.

# Example

```mlir
// Scalar tangent value.
%a = math.tan %b : f64
```
"""
function tan(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.tan", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`tanh`

The `tanh` operation computes the hyperbolic tangent. It takes one operand
of floating point type (i.e., scalar, tensor or vector) and returns one
result of the same type. It has no standard attributes.

# Example

```mlir
// Scalar hyperbolic tangent value.
%a = math.tanh %b : f64
```
"""
function tanh(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.tanh", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`trunc`

The `trunc` operation returns the operand rounded to the nearest integer
value in floating-point format. It takes one operand of floating point type
(i.e., scalar, tensor or vector) and produces one result of the same type.
The operation always rounds to the nearest integer not larger in magnitude
than the operand, regardless of the current rounding direction.

# Example

```mlir
// Scalar trunc operation.
%a = math.trunc %b : f64
```
"""
function trunc(operand::Value; result=nothing::Union{Nothing, IR.Type}, fastmath=nothing, location=Location())
    results = IR.Type[]
    operands = Value[operand, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(fastmath) && push!(attributes, namedattribute("fastmath", fastmath))
    
    create_operation(
        "math.trunc", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

end # math
