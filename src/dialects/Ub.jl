module Ub

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation
import ..Dialects: make_named_attribute

"""
poison

The `poison` operation materializes a compile-time poisoned constant value
to indicate deferred undefined behavior.
`value` attirbute is needed to indicate an optional additional poison
semantics (e.g. partially poisoned vectors), default value indicates results
is fully poisoned.

Syntax:

```
poison-op ::= `poison` (`<` value `>`)? `:` type
```

Examples:

```
// Short form
%0 = ub.poison : i32
// Long form
%1 = ub.poison <#custom_poison_elements_attr> : vector<4xi64>
```
  
"""
function Poison(; location::Location, result_::MLIRType, value_=nothing::Union{Nothing, Union{NamedAttribute, Attribute}})
  results = [result_]
  operands = []
  regions = []
  successors = []
  attributes = []

  (value_ != nothing) && push!(attributes, make_named_attribute("value", value_))

  create_operation(
        "ub.poison", location, 
        results = results, 
        operands = operands,
        owned_regions = regions, 
        successors = successors, 
        attributes = attributes,
        result_inference=false
      )
end


end #Ub
