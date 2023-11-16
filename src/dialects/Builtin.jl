module Builtin

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation
import ..Dialects: make_named_attribute

"""
module

A `module` represents a top-level container operation. It contains a single
[graph region](../LangRef.md#control-flow-and-ssacfg-regions) containing a single block
which can contain any operations and does not have a terminator. Operations
within this region cannot implicitly capture values defined outside the module,
i.e. Modules are [IsolatedFromAbove](../Traits.md#isolatedfromabove). Modules have
an optional [symbol name](../SymbolsAndSymbolTables.md) which can be used to refer
to them in operations.

Example:

```mlir
module {
  func.func @foo()
}
```
  
"""
function Module(; location::Location, bodyRegion_::Region, sym_name_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}}, sym_visibility_=nothing::Union{Nothing, Union{NamedAttribute, String}})
  results = []
  operands = []
  regions = [bodyRegion_]
  successors = []
  attributes = []

  (sym_name_ != nothing) && push!(attributes, make_named_attribute("sym_name", sym_name_))
  (sym_visibility_ != nothing) && push!(attributes, make_named_attribute("sym_visibility", sym_visibility_))

  create_operation(
        "builtin.module", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


"""
unrealized_conversion_cast

An `unrealized_conversion_cast` operation represents an unrealized
conversion from one set of types to another, that is used to enable the
inter-mixing of different type systems. This operation should not be
attributed any special representational or execution semantics, and is
generally only intended to be used to satisfy the temporary intermixing of
type systems during the conversion of one type system to another.

This operation may produce results of arity 1-N, and accept as input
operands of arity 0-N.

Example:

```mlir
// An unrealized 0-1 conversion. These types of conversions are useful in
// cases where a type is removed from the type system, but not all uses have
// been converted. For example, imagine we have a tuple type that is
// expanded to its element types. If only some uses of an empty tuple type
// instance are converted we still need an instance of the tuple type, but
// have no inputs to the unrealized conversion.
%result = unrealized_conversion_cast to !bar.tuple_type<>

// An unrealized 1-1 conversion.
%result1 = unrealized_conversion_cast %operand : !foo.type to !bar.lowered_type

// An unrealized 1-N conversion.
%results2:2 = unrealized_conversion_cast %tuple_operand : !foo.tuple_type<!foo.type, !foo.type> to !foo.type, !foo.type

// An unrealized N-1 conversion.
%result3 = unrealized_conversion_cast %operand, %operand : !foo.type, !foo.type to !bar.tuple_type<!foo.type, !foo.type>
```
  
"""
function UnrealizedConversionCast(; location::Location, outputs_::Vector{MLIRType}, inputs_::Vector{Value})
  results = [outputs_...]
  operands = [inputs_...]
  regions = []
  successors = []
  attributes = []
  
  create_operation(
        "builtin.unrealized_conversion_cast", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


end #Builtin
