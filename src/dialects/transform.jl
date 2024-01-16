module transform

import ...IR: NamedAttribute, MLIRType, get_value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`alternatives`

This op may have an arbitrary number of regions, each of which represents a
sequence of transform operations to be applied to the same payload IR. The
regions are visited in order of appearance, and transforms in them are
applied in their respective order of appearance. If one of these transforms
fails to apply, the remaining ops in the same region are skipped an the next
region is attempted. If all transformations in a region succeed, the
remaining regions are skipped and the entire \"alternatives\" transformation
succeeds. If all regions contained a failing transformation, the entire
\"alternatives\" transformation fails.

It is up to the nested operations to define which errors are \"recoverable\"
(or \"silenceable\") and allow another alternatives to be attempted, and which
errors should be propagated without attempting the other alternatives.

The single operand of this operation is the scope in which the alternative
transformation sequences are attempted, that is, an operation in the payload
IR that contains all the other operations that may be modified by the
transformations. The scope operation must be isolated from above. There is
no check that the transforms are indeed scoped as their \"apply\" methods can
be arbitrarily complex. Therefore it is the responsibility of the user to
ensure that the transforms are scoped correctly, or to produce an
irrecoverable error and thus abort the execution without attempting the
remaining alternatives. Note that the payload IR outside of the given scope
is not necessarily in the valid state, or even accessible to the
transformation.

The changes to the IR within the scope performed by transforms in the failed
alternative region are reverted before attempting the next region.
Practically, this is achieved by cloning the scope. Therefore it is advised
to limit the scope as much as possible and place the most likely
alternatives early in the region list. The operation is also isolated from
above and requires rediscovering the operations within the given scope to
avoid additional handle invalidation. The latter restriction may be lifted
in the future.

Each of the regions may yield transform IR handles. The handles of the first
successful alternative region are returned as the results of the
\"alternatives\" op. Therefore, each alternative region must yield the same
number of results, which should also match the number and the types of the
\"alternatives\" op results.

Remark: this op allows one to implement a simple \"try\" construct as follows:

```mlir
%result = transform.alternatives %scope {
^bb0(%arg0: !transform.any_op):
  // Try a fallible transformation.
  %0 = transform.fallible %arg0 // ...
  // If succeeded, yield the the result of the transformation.
  transform.yield %0 : !transform.any_op
}, {
^bb0(%arg0: !transform.any_op):
  // Otherwise, the second alternative is tried and it always succeeds by
  // returning the original handle.
  transform.yield %arg0 : !transform.any_op
}
```
"""
function alternatives(scope=nothing; results::Vector{MLIRType}, alternatives::Vector{Region}, location=Location())
    results = MLIRType[results..., ]
    operands = API.MlirValue[]
    owned_regions = Region[alternatives..., ]
    successors = Block[]
    attributes = NamedAttribute[]
    (scope != nothing) && push!(operands, get_value(scope))
    
    create_operation(
        "transform.alternatives", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`annotate`

Adds an attribute with the given `name` to the `target` operation. An
optional `param` handle can be provided to give the attribute a specific
value, else a UnitAttr is added. A single attribute will be broadcasted to
all target operations, otherwise the attributes will be mapped 1:1 based on
the order within the handles.

Fails silently if the length of the parameter payload does not match the length of
the target payload. Does not consume the provided handles.
"""
function annotate(target, param=nothing; name, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("name", name), ]
    (param != nothing) && push!(operands, get_value(param))
    
    create_operation(
        "transform.annotate", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_canonicalization`

This op populates all canonicalization patterns of all loaded dialects in
an `apply_patterns` transform.
"""
function apply_patterns_canonicalization(; location=Location())
    results = MLIRType[]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.apply_patterns.canonicalization", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_cse`

This transform applies common subexpression elimination (CSE) to the body
of the targeted op.

This transform reads the target handle and modifies the payload. Existing
handles to operations inside of the targeted op are retained and updated if
necessary. Note that this can lead to situations where a handle, that was
previously mapped to multiple distinct (but equivalent) operations, is now
mapped to the same operation multiple times.
"""
function apply_cse(target; location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.apply_cse", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_conversion_patterns`

This transform applies the specified conversion patterns to the targeted op
and all nested ops. By default, this transform applies a \"full\" dialect
conversion. If the `partial_conversion` unit attribute is present, this
transform applies a partial dialect conversion.

The patterns that should be applied are specified in the first graph region
of this op. They must implement the
`ConversionPatternDescriptorOpInterface`. The order in which patterns are
applied is unspecified; i.e., the ordering of ops in the region of this op
is irrelevant.

The second, optional graph region contains exactly one op that specifies
default type converter that should be used with this dialect conversion. If
provided, this op must implement the `TypeConverterBuilderOpInterface`.
Type converters are a property of conversion patterns: each conversion
pattern stores the type converter that should be used in its C++ class. Each
conversion pattern descriptor can optionally specify a type converter in its
`getTypeConverter` interface method. If no type converter is specified in
this method, the default type converter of the dialect conversion is used.
Default type converters are useful if the same type converter should be used
for multiple sets of conversion patterns. (Patterns that should not use this
default type converter specify their own type converter.)

The `legal_ops`, `illegal_ops`, `legal_dialects`, `illegal_dialects`
attributes specify the conversion target.

This transform consumes the `target` handle and modifies the payload. It
does not produce any handles.

This transform fails silently if the dialect conversion was unsuccessful.
"""
function apply_conversion_patterns(target; legal_ops=nothing, illegal_ops=nothing, legal_dialects=nothing, illegal_dialects=nothing, partial_conversion=nothing, patterns::Region, default_type_converter_region::Vector{Region}, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[patterns, default_type_converter_region..., ]
    successors = Block[]
    attributes = NamedAttribute[]
    (legal_ops != nothing) && push!(attributes, namedattribute("legal_ops", legal_ops))
    (illegal_ops != nothing) && push!(attributes, namedattribute("illegal_ops", illegal_ops))
    (legal_dialects != nothing) && push!(attributes, namedattribute("legal_dialects", legal_dialects))
    (illegal_dialects != nothing) && push!(attributes, namedattribute("illegal_dialects", illegal_dialects))
    (partial_conversion != nothing) && push!(attributes, namedattribute("partial_conversion", partial_conversion))
    
    create_operation(
        "transform.apply_conversion_patterns", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_dce`

This transform applies dead code elimination (DCE) to the body of the
targeted op.

Note: \"transform.apply_patterns\" with an empty region can also be used to
remove dead ops. However, that op applies additional simplifications such as
op folding and region simplification.

This transform reads the target handle and modifies the payload. Note that
this transform may silently remove payload ops from handles.
"""
function apply_dce(target; location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.apply_dce", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_licm`

This transform moves side-effect free, loop invariant code out of the
targeted loop-like op. The targeted op must implement the
`LoopLikeOpInterface`.

Note: To move invariant ops from a loop nest, this transform must be applied
to each loop of the loop nest, starting with the inner-most loop.

This transform reads the target handle and modifies the payload.
"""
function apply_licm(target; location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.apply_licm", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns`

This transform greedily applies the specified patterns to the body of the
targeted op until a fixpoint was reached. Patterns are not applied to the
targeted op itself.

The patterns that should be applied are specified in the graph region of
this op. They must implement the `PatternDescriptorOpInterface`. The order
in which patterns are applied is unspecified; i.e., the ordering of ops in
the region of this op is irrelevant.

If `apple_cse` is set, the greedy pattern rewrite is interleaved with
common subexpression elimination (CSE): both are repeated until a fixpoint
is reached.

This transform only reads the target handle and modifies the payload. If a
pattern erases or replaces a tracked op, the mapping is updated accordingly.

Only replacements via `RewriterBase::replaceOp` or `replaceOpWithNewOp` are
considered \"payload op replacements\". Furthermore, only if the replacement
values are defined by the same op and that op has the same type as the
original op, the mapping is updated. Otherwise, this transform fails
silently. More details can be found at the documentation site of
`TrackingListener`.

This transform also fails silently if the pattern application did not
converge within the default number of iterations/rewrites of the greedy
pattern rewrite driver.
"""
function apply_patterns(target; apply_cse=nothing, patterns::Region, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[patterns, ]
    successors = Block[]
    attributes = NamedAttribute[]
    (apply_cse != nothing) && push!(attributes, namedattribute("apply_cse", apply_cse))
    
    create_operation(
        "transform.apply_patterns", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_registered_pass`

This transform applies the specified pass or pass pipeline to the targeted
ops. The name of the pass/pipeline is specified as a string attribute, as
set during pass/pipeline registration. Optionally, pass options may be
specified as a string attribute. The pass options syntax is identical to the
one used with \"mlir-opt\".

This op first looks for a pass pipeline with the specified name. If no such
pipeline exists, it looks for a pass with the specified name. If no such
pass exists either, this op fails definitely.

This transform consumes the target handle and produces a new handle that is
mapped to the same op. Passes are not allowed to remove/modify the operation
that they operate on, so the target op is guaranteed to still exist. The
target handle is invalidated because a pass may arbitrarily modify the body
of targeted ops.
"""
function apply_registered_pass(target; result::MLIRType, pass_name, options=nothing, location=Location())
    results = MLIRType[result, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("pass_name", pass_name), ]
    (options != nothing) && push!(attributes, namedattribute("options", options))
    
    create_operation(
        "transform.apply_registered_pass", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_conversion_patterns_dialect_to_llvm`

Collects patterns that convert ops from the specified dialect to LLVM
dialect ops. These patterns require an \"LLVMTypeConverter\".

Note: Only dialects that implement the `ConvertToLLVMPatternInterface` are
supported. Any conversion target modifications by interface implementations
are currently ignored. The conversion target is fully specified by the
enclosing \"apply_conversion_patterns\" op.
"""
function apply_conversion_patterns_dialect_to_llvm(; dialect_name, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dialect_name", dialect_name), ]
    
    create_operation(
        "transform.apply_conversion_patterns.dialect_to_llvm", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`cast`

"""
function cast(input; output::MLIRType, location=Location())
    results = MLIRType[output, ]
    operands = API.MlirValue[get_value(input), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.cast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`foreach_match`

Given a pair of co-indexed lists of transform dialect symbols (such as
`transform.named_sequence`), walks the payload IR associated with the root
handle and interprets the symbols as matcher/action pairs by applying the
body of the corresponding symbol definition. The symbol from the first list
is the matcher part: if it results in a silenceable error, the error is
silenced and the next matcher is attempted. Definite failures from any
matcher stop the application immediately and are propagated unconditionally.
If none of the matchers succeeds, the next payload operation in walk order
(post-order at the moment of writing, double check `Operation::walk`) is
matched. If a matcher succeeds, the co-indexed action symbol is applied and
the following matchers are not applied to the same payload operation. If the
action succeeds, the next payload operation in walk order is matched. If it
fails, both silenceable and definite errors are propagated as the result of
this op.

The matcher symbol must take one operand of a type that implements the same
transform dialect interface as the `root` operand (a check is performed at
application time to see if the associated payload satisfies the constraints
of the actual type). It must not consume the operand as multiple matchers
may be applied. The matcher may produce any number of results. The action
symbol paired with the matcher must take the same number of arguments as the
matcher has results, and these arguments must implement the same transform
dialect interfaces, but not necessarily have the exact same type (again, a
check is performed at application time to see if the associated payload
satisfies the constraints of actual types on both sides). The action symbol
may not have results. The actions are expected to only modify payload
operations nested in the `root` payload operations associated with the
operand of this transform operation. Furhermore, the actions may not modify
operations outside of the currently matched payload operation, e.g., they
may not modify sibling or parent operations. If such behavior is desired,
the parent must be matched first and the nested operations obtained by
traversing the IR from the parent. This is due to the matching being
performed as a post-order IR walk.

This operation consumes the operand and produces a new handle associated
with the same payload. This is necessary to trigger invalidation of handles
to any of the payload operations nested in the payload operations associated
with the operand, as those are likely to be modified by actions. 

By default, the root payload operation associated with the operand is not
matched. This is to support the conservative case where applied actions may
invalidate the root payload operation. If the optional `restrict_root`
attribute is set, the root operand is guaranteed to not be invalidated by any
of the applied actions. In such cases, the root payload operation is also
matched. This is useful because matching the root payload operation is a
common idiom, when e.g. matching a func.func directly and operations nested
under it.

The operation succeeds if none of the matchers produced a definite failure
during application and if all of the applied actions produced success. Note
that it also succeeds if all the matchers failed on all payload operations,
i.e. failure to apply is not an error. The operation produces a silenceable
failure if any applied action produced a silenceable failure. In this case,
the resulting handle is associated with an empty payload. The operation
produces a definite failure if any of the applied matchers or actions
produced a definite failure.
"""
function foreach_match(root; updated::MLIRType, restrict_root=nothing, matchers, actions, location=Location())
    results = MLIRType[updated, ]
    operands = API.MlirValue[get_value(root), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("matchers", matchers), namedattribute("actions", actions), ]
    (restrict_root != nothing) && push!(attributes, namedattribute("restrict_root", restrict_root))
    
    create_operation(
        "transform.foreach_match", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`foreach`

This op has exactly one region with exactly one block (\"body\"). The body is
executed for each payload op that is associated to the target operand in an
unbatched fashion. I.e., the block argument (\"iteration variable\") is always
mapped to exactly one payload op.

This op always reads the target handle. Furthermore, it consumes the handle
if there is a transform op in the body that consumes the iteration variable.
This op does not return anything.

The transformations inside the body are applied in order of their
appearance. During application, if any transformation in the sequence fails,
the entire sequence fails immediately leaving the payload IR in potentially
invalid state, i.e., this operation offers no transformation rollback
capabilities.

This op generates as many handles as the terminating YieldOp has operands.
For each result, the payload ops of the corresponding YieldOp operand are
merged and mapped to the same resulting handle.
"""
function foreach(target; results::Vector{MLIRType}, body::Region, location=Location())
    results = MLIRType[results..., ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.foreach", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`get_consumers_of_result`

The handle defined by this Transform op corresponds to all operations that
consume the SSA value defined by the `target` and `result_number`
arguments.
This operation applies to a single payload operation, otherwise it 
definitely fails.
The return handle points to the consuming operations operations, which can
be empty.
"""
function get_consumers_of_result(target; consumers::MLIRType, result_number, location=Location())
    results = MLIRType[consumers, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("result_number", result_number), ]
    
    create_operation(
        "transform.get_consumers_of_result", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`get_defining_op`

The handle defined by this Transform op corresponds to the defining op of
the targeted value.

This transform fails silently if the targeted value is a block argument.
"""
function get_defining_op(target; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.get_defining_op", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`get_parent_op`

The handle defined by this Transform op corresponds to the parents of the
targeted payload ops (in the same order).

Requirements that parent ops must fulfill can be optionally specified. In
that case for each target op, the closest parent op that fulfills all
requirements, is returned.
- `isolated_from_above`: the parent op must be isolated from above
- `allow_empty_results`: get_parent_op is allowed to return an empty list
  and still succeeds. In such a case, if get_parent_op fails for any
  operation in the list, the entire transform returns an empty handle.
- `op_name`: the parent op must have the specified name
- `nth_parent`: get the n-th parent of that satisfies the above requirements

If `deduplicate` is set, the result handle does not contain any duplicate
ops. For example, given the list
\"(childof(A), childof(B), childof(B), childof(A), childof(B))\", the
resulting list will be just \"(A, B)\". Note that no other semantic ordering
is applied, e.g., \"B\" may itself be a parent of \"A\". This may have an impact
on the further transformation applied to the handle produced here.

If any of the given Payload IR ops has no such suitable parent, then:
  - if `allow_empty_results` is set, the result handle is empty
  - otherwise, the transformation produces a silenceable failure.
"""
function get_parent_op(target; parent::MLIRType, isolated_from_above=nothing, allow_empty_results=nothing, op_name=nothing, deduplicate=nothing, nth_parent=nothing, location=Location())
    results = MLIRType[parent, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (isolated_from_above != nothing) && push!(attributes, namedattribute("isolated_from_above", isolated_from_above))
    (allow_empty_results != nothing) && push!(attributes, namedattribute("allow_empty_results", allow_empty_results))
    (op_name != nothing) && push!(attributes, namedattribute("op_name", op_name))
    (deduplicate != nothing) && push!(attributes, namedattribute("deduplicate", deduplicate))
    (nth_parent != nothing) && push!(attributes, namedattribute("nth_parent", nth_parent))
    
    create_operation(
        "transform.get_parent_op", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`get_producer_of_operand`

The handle defined by this Transform op corresponds to operation that
produces the SSA value defined by the `target` and `operand_number`
arguments. If the origin of the SSA value is not an operations (i.e. it is
a block argument), the transform silently fails.
The return handle points to only the subset of successfully produced
computational operations, which can be empty.
"""
function get_producer_of_operand(target; producer::MLIRType, operand_number, location=Location())
    results = MLIRType[producer, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("operand_number", operand_number), ]
    
    create_operation(
        "transform.get_producer_of_operand", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`get_result`

The handle defined by this Transform op corresponds to the OpResult with
`result_number` that is defined by the given `target` operation.

This transform fails silently if the targeted operation does not have enough
results. It reads the target handle and produces the result handle.
"""
function get_result(target; result::MLIRType, result_number, location=Location())
    results = MLIRType[result, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("result_number", result_number), ]
    
    create_operation(
        "transform.get_result", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`get_type`

This operation creates a new Transform parameter containing the
type(s) of the value(s) associated with the operand handle.

This transform never fails.
"""
function get_type(value; type_param::MLIRType, elemental=nothing, location=Location())
    results = MLIRType[type_param, ]
    operands = API.MlirValue[get_value(value), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (elemental != nothing) && push!(attributes, namedattribute("elemental", elemental))
    
    create_operation(
        "transform.get_type", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`include`

The application of this transform operation is equivalent to applying the
operations contained in the named transform sequence with operands being
remapped to block arguments. The behavior of the operation when a
transformation in the included named sequence produces a silenceable error
is controlled by the `failure_propagation_mode` attribute. When set to
`propagate`, the failure of any nested transformation in the sequence
implies immediate failure of the entire sequence with a silenceable error,
and no further transformation is attempted. When set to `suppress`,
silenceable errors in nested operations are ignored and further
transformations are applied. Beware that even silenceable errors may leave
the payload IR in a state unsuitable for further transformations. It is the
responsibility of the user to ensure the following transformations are
robust enough when errors are suppressed. Definite errors are propagated
immediately regardless of the mode. The objects associated with the results
of this operation are the same as those associated with the operands of the
`transform.yield` in the referenced named sequence.
"""
function include_(operands; results::Vector{MLIRType}, target, failure_propagation_mode, location=Location())
    results = MLIRType[results..., ]
    operands = API.MlirValue[get_value.(operands)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("target", target), namedattribute("failure_propagation_mode", failure_propagation_mode), ]
    
    create_operation(
        "transform.include", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`match_operation_empty`

Succeeds if the handle is not associated to any op.
"""
function match_operation_empty(operand_handle; location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value(operand_handle), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.match.operation_empty", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`match_operation_name`

Succeeds if the operation associated with the operand handle has one of the
given operation names. Produces a silenceable failure otherwise.

If more than one payload operation is associated with the operand handle,
produces a definite failure.
"""
function match_operation_name(operand_handle; op_names, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value(operand_handle), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("op_names", op_names), ]
    
    create_operation(
        "transform.match.operation_name", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`match_param_cmpi`

Succeeds if all of the co-indexed values associated with the given
parameters relate as specified by the predicate (greater than, less than,
equal to, or their combinations). Comparison treats all values as signed.
Produces a silenceable failure otherwise.
"""
function match_param_cmpi(param, reference; predicate, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value(param), get_value(reference), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("predicate", predicate), ]
    
    create_operation(
        "transform.match.param.cmpi", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`merge_handles`

Creates a new Transform IR handle value that points to the same Payload IR
operations/values/parameters as the operand handles. The Payload IR elements
are listed in the same order as they are in the operand handles, grouped by
operand handle, e.g., all Payload IR associated with the first handle comes
first, then all Payload IR associated with the second handle and so on. If
`deduplicate` is set, do not add the given Payload IR operation, value, or
parameter more than once to the final list regardless of it coming from the
same or different handles. Consumes the operands and produces a new handle.
"""
function merge_handles(handles; result=nothing::Union{Nothing, MLIRType}, deduplicate=nothing, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value.(handles)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result != nothing) && push!(results, result)
    (deduplicate != nothing) && push!(attributes, namedattribute("deduplicate", deduplicate))
    
    create_operation(
        "transform.merge_handles", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`named_sequence`

Defines a named (callable, function-like) sequence of other Transform
dialect operations that can be included using `transform.include` as part of
another Transform dialect construct. This sequence is not processed
immediately but rather dispatched to when the inclusion is processed. The
arguments and results can be used to communicate a subset of mapping into
the named sequence. The sequence must consist of a single block and end with
a `transform.yield` terminator. The operands of the terminator become the
results of the `transform.include`.

When dispatched to, the operations in the named sequence are executed one by
one, similarly to the regular unnamed sequence. The failure propagation mode
is specified on the `transform.include`. Different inclusions may use
different failure propagation modes. This transform operation always
succeeds by itself, but the inclusion may fail if any of the operations
fail.

Named sequences can only appear at the top-level of the Transform dialect
nesting structure. That is, they cannot be nested in other Transform dialect
operations. Furthermore, one of the ancestors must have the `SymbolTable`
trait and have the `transform.with_named_sequence` attribute attached.

Named sequences may include other named sequences via `transform.include`,
but recursion is *not* allowed.
"""
function named_sequence(; sym_name, function_type, sym_visibility=nothing, arg_attrs=nothing, res_attrs=nothing, body::Region, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), namedattribute("function_type", function_type), ]
    (sym_visibility != nothing) && push!(attributes, namedattribute("sym_visibility", sym_visibility))
    (arg_attrs != nothing) && push!(attributes, namedattribute("arg_attrs", arg_attrs))
    (res_attrs != nothing) && push!(attributes, namedattribute("res_attrs", res_attrs))
    
    create_operation(
        "transform.named_sequence", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`num_associations`

Given an argument, handle or parameter, returns a new parameter associated
with a single 64-bit number that corresponds to the number of payload
objects (operations or values for a handle, attributes for a parameter)
associated with the argument.

Always succeeds.
"""
function num_associations(handle; num::MLIRType, location=Location())
    results = MLIRType[num, ]
    operands = API.MlirValue[get_value(handle), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.num_associations", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`param_constant`

Produces a new transform dialect parameter associated with the singleton
list containing the given attribute. The operation itself always succeeds,
but the general association check may fail if the parameter type does not
accept the given kind of attribute as valid.
"""
function param_constant(; param::MLIRType, value, location=Location())
    results = MLIRType[param, ]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("value", value), ]
    
    create_operation(
        "transform.param.constant", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`print`

This op dumps each payload op that is associated with the `target` operand
to stderr. It also prints the `name` string attribute. If no target is
specified, the top-level op is dumped.

This op is useful for printf-style debugging.
"""
function print(target=nothing; name=nothing, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (target != nothing) && push!(operands, get_value(target))
    (name != nothing) && push!(attributes, namedattribute("name", name))
    
    create_operation(
        "transform.print", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`replicate`

Produces a new handle associated with a list of payload IR ops that is
computed by repeating the list of payload IR ops associated with the
operand handle as many times as the \"pattern\" handle has associated
operations. For example, if pattern is associated with [op1, op2] and the
operand handle is associated with [op3, op4, op5], the resulting handle
will be associated with [op3, op4, op5, op3, op4, op5].

This transformation is useful to \"align\" the sizes of payload IR lists
before a transformation that expects, e.g., identically-sized lists. For
example, a transformation may be parameterized by same notional per-target
size computed at runtime and supplied as another handle, the replication
allows this size to be computed only once and used for every target instead
of replicating the computation itself.

Note that it is undesirable to pass a handle with duplicate operations to
an operation that consumes the handle. Handle consumption often indicates
that the associated payload IR ops are destroyed, so having the same op
listed more than once will lead to double-free. Single-operand
MergeHandlesOp may be used to deduplicate the associated list of payload IR
ops when necessary. Furthermore, a combination of ReplicateOp and
MergeHandlesOp can be used to construct arbitrary lists with repetitions.
"""
function replicate(pattern, handles; replicated::Vector{MLIRType}, location=Location())
    results = MLIRType[replicated..., ]
    operands = API.MlirValue[get_value(pattern), get_value.(handles)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.replicate", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`select`

The handle defined by this Transform op corresponds to all operations among
`target` that have the specified properties. Currently the following
properties are supported:

- `op_name`: The op must have the specified name.

The result payload ops are in the same relative order as the targeted ops.
This transform op reads the `target` handle and produces the `result`
handle. It reads the payload, but does not modify it.
"""
function select(target; result::MLIRType, op_name, location=Location())
    results = MLIRType[result, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("op_name", op_name), ]
    
    create_operation(
        "transform.select", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`sequence`

The transformations indicated by the sequence are applied in order of their
appearance. Each value produced by a transformation within the sequence
corresponds to a group of operations or values in the payload IR, or to a
group of parameters, depending on the type of the value. The behavior of the
operation when a nested transformation produces a silenceable error is
controlled by the `failure_propagation_mode` attribute. When set to
`propagate`, the failure of any nested transformation in the sequence
implies immediate failure of the entire sequence with a silenceable error,
and no further transformation is attempted. When set to `suppress`,
silenceable errors in nested operations are ignored and further
transformations are applied. Beware that even silenceable errors may leave
the payload IR in a state unsuitable for further transformations. It is the
responsibility of the caller to ensure the following transformations are
robust enough when errors are suppressed. Definite errors reported by nested
transformations abort the sequence regardless of the propagation mode. The
set of modes may be extended in the future, e.g., to collect silenceable
errors and report them after attempting all transformations in the sequence.

The entry block of this operation has a single argument that maps to either
the operand if provided or the top-level container operation of the payload
IR, typically the root operation of the pass interpreting the transform
dialect. Operand omission is only allowed for sequences not contained in
another sequence.

The type of the block argument must match the type of the operand. If the
sequence is a top-level transform (without an operand), it can be used for
matching operations if the specified type within the top-level container
payload IR (including the container op itself). E.g.:

```mlir
transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  // %arg1 is mapped to the top-level container of the payload IR, which is
  // typically a module
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.op<\"func.func>\"):
  // %arg1 is mapped to all \"func.func\" ops within and including the
  // top-level container of the payload IR. Nested operations that have the
  // specified op type are not included.
}
```

The body of the sequence terminates with an implicit or explicit
`transform.yield` op. The operands of the terminator are returned as the
results of the sequence op.
"""
function sequence(root=nothing; extra_bindings, results::Vector{MLIRType}, failure_propagation_mode, body::Region, location=Location())
    results = MLIRType[results..., ]
    operands = API.MlirValue[get_value.(extra_bindings)..., ]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("failure_propagation_mode", failure_propagation_mode), ]
    (root != nothing) && push!(operands, get_value(root))
    push!(attributes, operandsegmentsizes([(root==nothing) ? 0 : 1length(extra_bindings), ]))
    
    create_operation(
        "transform.sequence", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`split_handle`

Splits `handle` into one or multiple handles, as specified by the number
of results of this operation. `handle` should be mapped to as many payload
ops as there are results. Otherwise, this transform will fail silently by
default. Each result handle is mapped to exactly one payload op. The order
of the payload ops is preserved, i.e., the i-th payload op is mapped to the
i-th result handle.

This operation is useful for ensuring a statically known number of
operations are tracked by the source `handle` and to extract them into
individual handles that can be further manipulated in isolation.

If there are more payload ops than results, the remaining ops are mapped to
the result with index `overflow_result`. If no `overflow_result` is
specified, the transform fails silently.

If there are fewer payload ops than results, the transform fails silently
if `fail_on_payload_too_small` is set to \"true\". Otherwise, it succeeds and
the remaining result handles are not mapped to any op. It also succeeds if
`handle` is empty and `pass_through_empty_handle` is set to \"true\",
regardless of `fail_on_payload_too_small`.
"""
function split_handle(handle; results::Vector{MLIRType}, pass_through_empty_handle=nothing, fail_on_payload_too_small=nothing, overflow_result=nothing, location=Location())
    results = MLIRType[results..., ]
    operands = API.MlirValue[get_value(handle), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (pass_through_empty_handle != nothing) && push!(attributes, namedattribute("pass_through_empty_handle", pass_through_empty_handle))
    (fail_on_payload_too_small != nothing) && push!(attributes, namedattribute("fail_on_payload_too_small", fail_on_payload_too_small))
    (overflow_result != nothing) && push!(attributes, namedattribute("overflow_result", overflow_result))
    
    create_operation(
        "transform.split_handle", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`verify`

This transform verifies the targeted ops. If at least one op fails to
verify, the transform fails definitely.

Note: This op was designed for debugging purposes and should be used like an
assertion. It is intentional that this op produces a definite failure and
not a silenceable one. Correctness of the program should not depend on this
op.

This transform reads the target handle.
"""
function verify(target; location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.verify", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`yield`

This terminator operation yields operation handles from regions of the
transform IR ops back to the containing op. It is not itself associated with
any transformation on the payload IR and is used for flow purposes only.
"""
function yield(operands; location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value.(operands)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.yield", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_linalg_erase_unnecessary_inputs`

Collects patterns that promote inputs to outputs and remove unused inputs of
`linalg.generic` ops.
"""
function apply_patterns_linalg_erase_unnecessary_inputs(; location=Location())
    results = MLIRType[]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.apply_patterns.linalg.erase_unnecessary_inputs", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_linalg_fold_unit_extent_dims_via_reshapes`

Collects patterns to fold unit-extent dimensions in operands/results of
linalg ops on tensors via reassociative reshape ops.
"""
function apply_patterns_linalg_fold_unit_extent_dims_via_reshapes(; location=Location())
    results = MLIRType[]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_linalg_fold_unit_extent_dims_via_slices`

Collects patterns to fold unit-extent dimensions in operands/results of
linalg ops on tensors via rank-reducing slices.
"""
function apply_patterns_linalg_fold_unit_extent_dims_via_slices(; location=Location())
    results = MLIRType[]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_patterns_linalg_tiling_canonicalization`

Collects canonicalization patterns relevant to apply after tiling patterns.
"""
function apply_patterns_linalg_tiling_canonicalization(; location=Location())
    results = MLIRType[]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.apply_patterns.linalg.tiling_canonicalization", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_bufferize_to_allocation`

This transform bufferizes the targeted operation and materializes the
result in a new allocation. It replaces all original uses of the target
result with the newly allocated buffer, wrapped in a
`bufferization.to_tensor` op. It returns a handle to the newly allocated
buffer. Furthermore, it returns a handle that is mapped to all newly created
ops.

Only bufferizable ops are that bufferize to a memory write or have an
aliasing OpOperand (and do not themselves bufferize to an allocation) are
supported. They are bufferized using their BufferizableOpInterface
implementation. E.g.:

```
%0 = tensor.insert %f into %dest[%pos] : tensor<10xf32>
```

Is bufferized to:

```
%alloc = memref.alloc() : memref<10xf32>
bufferization.materialize_in_destination %dest in %alloc
memref.store %f, %alloc[%pos] : memref<10xf32>
%0 = bufferization.to_tensor %alloc restrict writable : memref<10xf32>
```

Selected ops that bufferize to an allocation (or need special handling) are
also supported:
- `tensor.pad` is lowered to an allocation, followed by a `linalg.fill` and
  and a buffer copy (all on memrefs).
- `vector.mask` is bufferized together with its region. The allocation is
  placed in front of the `vector.mask` op.

An optional memory space attribute can be specified for the materialized
buffer allocation.

If a memory copy is needed, a \"bufferization.materialize_in_destination\" is
used when possible. This is an op with tensor semantics that will bufferize
to a memory copy later. Which concrete op will be used for the memory copy
is up to the bufferization framework. Alternatively, a custom memcpy op can
be specified via `memcpy_op`. Currently supported are \"memref.copy\" and
\"linalg.copy\". In that case, the source of each memcpy must not have a
custom memory space. Furthermore, because the future buffer layout unknown
for a given tensor, a fully dynamic layout is assumed for best
compatibility. Users should use \"bufferization.materialize_in_destination\"
when possible.

\"memref.alloc\" is used for new buffer allocations. The buffer is deallocated
at the end of the block if the \"emit_dealloc\" attribute is present. If this
attribute is not present, the allocated memory will be leaked. However,
running the `-buffer-deallocation-pipeline` after all bufferization is done
will properly insert the corresponding deallocation(s). Custom allocation
ops can be specified via `alloc_op`. Currently supported are \"memref.alloc\"
and \"memref.alloca\". In case of a \"memref.alloca\", the buffer is not
deallocated.

If `bufferize_destination_only` is set, only the destination operands of the
op are bufferized to a new memory allocation, but not the op itself.

#### Return modes

This operation consumes the `target` handle and produces the
`allocated_buffer` and `new_ops` handles. It always succeeds.
"""
function structured_bufferize_to_allocation(target; allocated_buffer::MLIRType, new_ops::MLIRType, memory_space=nothing, memcpy_op=nothing, alloc_op=nothing, bufferize_destination_only=nothing, emit_dealloc=nothing, location=Location())
    results = MLIRType[allocated_buffer, new_ops, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (memory_space != nothing) && push!(attributes, namedattribute("memory_space", memory_space))
    (memcpy_op != nothing) && push!(attributes, namedattribute("memcpy_op", memcpy_op))
    (alloc_op != nothing) && push!(attributes, namedattribute("alloc_op", alloc_op))
    (bufferize_destination_only != nothing) && push!(attributes, namedattribute("bufferize_destination_only", bufferize_destination_only))
    (emit_dealloc != nothing) && push!(attributes, namedattribute("emit_dealloc", emit_dealloc))
    
    create_operation(
        "transform.structured.bufferize_to_allocation", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_convert_conv2d_to_img2col`

Convert linalg.conv_2d_xxx into linalg.generic (for img2col packing)
and linalg.matmul.

A convolution operation can be written as a matrix-matrix multiplication by
unfolding the cross-correlation between input and filter and explicitly copy
overlapped sliding window inputs.

Consider 2D input X with single channel input and output and 2x2 filter W:
```
[x(0, 0)  , x(0, 1)  , ...,   x(0, n)  ]
[x(1, 0)  , x(1, 1)  , ...,   x(1, n)  ]
[.        ,  .       ,.   ,      .     ]            [w(0, 0), w(0, 1)]
[.        ,  .       , .  ,      .     ]    (conv)  [w(1, 0), w(1, 1)]
[.        ,  .       ,   .,      .     ]
[x(n-1, 0), x(n-1, 1), ..., x(n-1, n-1)]
```

The packed input data (img2col) is a matrix with |rows| = output spatial
size, |columns| = filter spatial size. To compute the output Y(i, j) we need
to calculate the dot product between filter window at input X(x, y)) and the
filter which will look like the following where r.h.s is the img2col matrix
and l.h.s is the flattned filter:
```
[x(0,0), x(0,1), x(1,0), x(1,1)]
[x(0,1), x(1,1), x(0,2), x(1,2)] (matmul) [w(0,0), w(0,1), w(1,0), w(1,1)]
[x(0,1), x(1,1), x(0,2), x(1,2)]
[   .  ,    .  ,    .  ,    .  ]
```

In general for 2D case with (N, H, W, C) input and (Kh, Kw, C, D) filter
and output (N, Ho, Wo, D) the convolution is the following matrix-matrix
multiplication (Ho x Wo, Kh x Kw x C) * (Kh x Kw x C, D) for each input in
the N input. For the case where N > 1 its a batched matrxi-matrix
multplication.

Returns two handles:
- One on the operation that produces the img2col tensor.
- One on the final operation of the sequence that replaces the original
  convolution.

#### Return modes:

Returns a definite failure if target is not isolated from above.
Returns a silenceable failure if the pattern application failed.
"""
function structured_convert_conv2d_to_img2col(target; img2col_tensor::MLIRType, transformed::MLIRType, location=Location())
    results = MLIRType[img2col_tensor, transformed, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.structured.convert_conv2d_to_img2col", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_decompose_interface`

TODO
"""
function structured_decompose_interface(target; transformed::MLIRType, location=Location())
    results = MLIRType[transformed, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.structured.decompose_interface", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_decompose`

Decomposes named complex operations, such as higher-dimensional
(depthwise) convolutions, into combinations of lower-dimensional equivalents
when possible.

#### Return modes

This operation ignores non-Linalg ops and drops them in the return.
If all the operations referred to by the `target` handle decompose
properly, the transform succeeds. Otherwise the transform silently fails.
The return handle points to only the subset of successfully produced
computational operations, which can be empty.
"""
function structured_decompose(target; transformed::MLIRType, location=Location())
    results = MLIRType[transformed, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.structured.decompose", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_eliminate_empty_tensors`

Try to eliminate all `tensor.empty` op uses that are anchored on a LinalgOp
within the targeted op.

This op is similar to `bufferization.eliminate_empty_tensors`, but specific
to LinalgOps.

`tensor.empty` ops cannot be bufferized. They can either be converted to
`bufferization.alloc_tensor` or replaced with another tensor (via this
transform). `tensor.empty` does not specify the contents of the returned
tensor so their results can be replaced with arbitrary tensor values as long
as the dimensions match.

This transform looks for `tensor.empty` ops where the SSA use-def chain of
the result ends in a supported LinalgOp (always following the aliasing
OpOperand/OpResult chain). The following LinalgOps are supported:
- Only parallel iterator types.
- The use-def chain ends in an input operand of the LinalgOp.
- The LinalgOp has an unused output operand with the same shape and
  indexing map.

# Example

```
%0 = tensor.empty()
%1 = linalg.matmul ins(...) outs(%0)
%2 = linalg.generic ins(%1) outs(%dest) {
  ^bb0(%in: f32, %out: f32):
  // out not used
}
```

Is rewritten with:
```
%0 = tensor.empty()
%1 = linalg.matmul ins(...) outs(%dest)
%2 = linalg.generic ins(%0) outs(%1) {
  ^bb0(%in: f32, %out: f32):
  // Use %out instead of %in
}
```

After this transformation, the \"ins\" operand has no uses inside the body of
the LinalgOp and can be folded away with existing cleanup patterns.
Afterwards, the tensor::EmptyOp can also fold away, so that the example can
bufferize without an allocation (in the absence of other conflicts).

#### Return modes

This transform reads the target handle and modifies the payload. It does
not produce any handle.
"""
function structured_eliminate_empty_tensors(target; location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.structured.eliminate_empty_tensors", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_fuse_into_containing_op`

Fuses the `producer_op` into the `containing_op`.
Returns a handle to the fused ops and the `new_containing_op`.

The producer is typically a slice of a tileable op (i.e., implements
TilingInterface). In that case, this transform computes the accessed
producer slice inside of the containing op (\"tile and fuse\") and if required,
creates a new containing op with outputs from the fused producer. Otherwise,
the entire producer is cloned inside the containing op (\"clone and fuse\").

The containing op handle must be associated with exactly one payload op. The
producer op handle may be associated with multiple payload ops. This
transform fuses producers one-by-one, always picking an unspecified producer
that has at least one use inside the containing op among the
producers. A producer can be listed multiple times in the handle.

Note: If a producer has multiple uses inside the containing op, it is
currently tiled and/or cloned multiple times into the containing op.
TODO: Reuse already fused OpResults instead of tiling/cloning a second time
when possible. Fuse producers according to a topological sorting to achieve
the largest amount of reuse.

#### Return modes

If at least one producer could not be fused, this operation fails silently.
This is the case when tiling fails or when no producer op could be found
among the remaining producers that has at least one use within the
containing op. I.e., \"producers\" that are not consumed within the containing
op are rejected by this operation.

This operation consumes the producer handle.
This operation only reads the containing op handle.
"""
function structured_fuse_into_containing_op(producer_op, containing_op; fused_op::MLIRType, new_containing_op::MLIRType, location=Location())
    results = MLIRType[fused_op, new_containing_op, ]
    operands = API.MlirValue[get_value(producer_op), get_value(containing_op), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.structured.fuse_into_containing_op", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_fuse`

Tiles the operations pointed to by the target handle and fuses their
producers greedily using the options provided as attributes.
"""
function structured_fuse(target; transformed::MLIRType, loops::Vector{MLIRType}, tile_sizes=nothing, tile_interchange=nothing, location=Location())
    results = MLIRType[transformed, loops..., ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (tile_sizes != nothing) && push!(attributes, namedattribute("tile_sizes", tile_sizes))
    (tile_interchange != nothing) && push!(attributes, namedattribute("tile_interchange", tile_interchange))
    
    create_operation(
        "transform.structured.fuse", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_generalize`

Transforms a named structured operation into the generic form with the
explicit attached region.

#### Return modes

This operation ignores non-Linalg ops and drops them in the return.
If all the operations referred to by the `target` handle generalize
properly, the transform succeeds. Otherwise the transform silently fails.
The return handle points to only the subset of successfully produced
equivalent generic operations, which can be empty or contain the original
ops if they were already in generic form.
"""
function structured_generalize(target; transformed::MLIRType, location=Location())
    results = MLIRType[transformed, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.structured.generalize", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_hoist_pad_build_packing_loop_nest`

Helper transform used to hoist a tensor.pad target operation. This operation
creates the packing loop nest required by the hoist_pad operation and makes
that functionality available independently.

TODO: In the future, we should consider rewriting as a tensor.pack after
hoisting since this abstraction is now available.

#### Return modes

This operation ignores non-tensor.pad ops and drops them in the result.
If any non-tensor.pad is passed, the transform emits a silenceable failure.

The return handle points to only the subset of successfully created packing
loop nests, which can be empty.
"""
function structured_hoist_pad_build_packing_loop_nest(target, loop; packing_loop::MLIRType, transpose=nothing, location=Location())
    results = MLIRType[packing_loop, ]
    operands = API.MlirValue[get_value(target), get_value(loop), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (transpose != nothing) && push!(attributes, namedattribute("transpose", transpose))
    
    create_operation(
        "transform.structured.hoist_pad.build_packing_loop_nest", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_hoist_pad`

Hoist the tensor.pad target operation by at most the given number of loops.
Optionally apply the transpose attribute to the inner dimensions.

TODO: In the future, we should consider rewriting as a tensor.pack after
hoisting since this abstraction is now available.
TODO: Maybe also return the linalg.generic transpose created at some point.

#### Return modes

This operation ignores non-tensor.pad ops and drops them in the result.
If any non-tensor.pad is passed, the transform emits a silenceable failure.

If all the operations referred to by the `target` handle padproperly, the
transform succeeds. Otherwise the transform silently fails.

The return handle points to only the subset of successfully hoisted
tensor.pad operations, which can be empty.
"""
function structured_hoist_pad(target; transformed::MLIRType, num_loops, transpose=nothing, location=Location())
    results = MLIRType[transformed, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("num_loops", num_loops), ]
    (transpose != nothing) && push!(attributes, namedattribute("transpose", transpose))
    
    create_operation(
        "transform.structured.hoist_pad", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_hoist_redundant_vector_transfers`

Hoist vector.transfer_read / vector.transfer_write pairs out of immediately
enclosing scf::ForOp iteratively, if the following conditions are true:
   1. The 2 ops access the same memref with the same indices.
   2. All operands are invariant under the enclosing scf::ForOp.
   3. No uses of the memref either dominate the transfer_read or are
   dominated by the transfer_write (i.e. no aliasing between the write and
   the read across the loop)

WARNING: This hoisting does not model parallelism and is generally incorrect
when used on distributed loops with memref semantics!
TODO: obsolete and should be retired.

#### Return modes:

The operation always succeeds and returns a handle to the transformed
function op.
"""
function structured_hoist_redundant_vector_transfers(target; transformed::MLIRType, location=Location())
    results = MLIRType[transformed, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.structured.hoist_redundant_vector_transfers", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_insert_slice_to_copy`

Targeted rewrite of an tensor.insert_slice to linalg.copy.
This is useful to materialize copies explicitly before bufferization and
transform them, avoiding the need to rediscover them after bufferization.

If the insert_slice source is already a linalg.copy, only return the source
op (i.e. do not create an additional linalg.copy op).

#### Return modes:

The operation always succeeds and returns a handle to the relevant
linalg.copy op.
"""
function structured_insert_slice_to_copy(target; transformed::MLIRType, location=Location())
    results = MLIRType[transformed, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.structured.insert_slice_to_copy", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_interchange`

Interchanges the iterators of the operations pointed to by the target handle
using the iterator interchange attribute.

#### Return modes

This operation ignores non-linalg::Generic ops and drops them in the return.
This operation fails if the interchange attribute is invalid.
If all the operations referred to by the `target` handle interchange
properly, the transform succeeds.
If any interchange fails, the transform definitely fails.
The return handle points to only the subset of successfully produced
interchanged operations, which can be empty.
"""
function structured_interchange(target; transformed::MLIRType, iterator_interchange=nothing, location=Location())
    results = MLIRType[transformed, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (iterator_interchange != nothing) && push!(attributes, namedattribute("iterator_interchange", iterator_interchange))
    
    create_operation(
        "transform.structured.interchange", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_lower_pack`

Rewrite a tensor.pack into tensor.pad + tensor.expand_shape + linalg.transpose.

#### Return modes

This operation ignores non-pack ops and drops them in the return.
This operation produces a silenceableFailure if the rewrite fails for any
reason.
If all the operations referred to by the `target` are rewritten, the
transform succeeds.
Return handles to the newly produced pad, expand_shape and transpose ops.
"""
function structured_lower_pack(target; pad_op::MLIRType, expand_shape_op::MLIRType, transpose_op::MLIRType, location=Location())
    results = MLIRType[pad_op, expand_shape_op, transpose_op, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.structured.lower_pack", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_lower_unpack`

Lower a tensor.unpack into empty + linalg.transpose + tensor.collapse_shape +
tensor.extract_slice.

#### Return modes

This operation ignores non-unpack ops and drops them in the return.
This operation produces a silenceableFailure if the rewrite fails for any
reason.
If all the operations referred to by the `target` are rewritten, the
transform succeeds.
Return handles to the newly produced empty, transpose, collapse_shape and extract_slice ops.
"""
function structured_lower_unpack(target; empty_op::MLIRType, transpose_op::MLIRType, collapse_shape_op::MLIRType, extract_slice_op::MLIRType, location=Location())
    results = MLIRType[empty_op, transpose_op, collapse_shape_op, extract_slice_op, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.structured.lower_unpack", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_gpu_map_copy_to_threads`

Targeted mapping of a linalg.copy / tensor.pad operation on tensors to a GPU
thread mapping.

This operation implements a greedy heuristic that determines a good
distribution of threads to break down the copy/pad operation into.
The heuristic is driven by considerations related to the underlying
architecture for which good high-level decisions are needed assuming certain
hardware features. Relevant features are exposed via first-class attributes
to control the behavior of the transformation at a high level.

For now, a single heuristic is implemented and can be extended on a per-need
basis.

#### Return modes

This operation fails definitely if there is an unsupported op (i.e., not
linalg.copy / tensor.pad) among the targeted op. Otherwise, the operation
always succeeds and returns a handle to the relevant tiled linalg.copy /
tensor.pad op and the enclosing scf.forall op.
"""
function structured_gpu_map_copy_to_threads(target; forall_op::MLIRType, tiled_op::MLIRType, total_num_threads, desired_bit_alignment, location=Location())
    results = MLIRType[forall_op, tiled_op, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("total_num_threads", total_num_threads), namedattribute("desired_bit_alignment", desired_bit_alignment), ]
    
    create_operation(
        "transform.structured.gpu.map_copy_to_threads", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_match`

Match op with the specified constraints, within the target op.

The following constraints are supported:
  - interface: an optional MatchInterfaceEnum specifying an enum
    representation for an interface to target.
  - ops: an optional StrArrayAttr specifying the concrete name of an op.
    Multiple names can be specified. Matched ops must have one of specified
    names.
  - attribute: the matched op must have all specified attributes (with their
    specified values).
  - filter_result_type: the matched op must return exactly this one type.
  - filter_operand_types: all the operands of the matched op must must be of
    this type. If more than a type is specified, then the length of the list
    must be equal to the number of operands in the matched op, and the match
    will succeed only if the operand types match all the types in the list
    in the order in which they are specified.

Note: Only ops that satisfy all specified constraints are matched.

TODO: Extend with regions to allow a limited form of constraints.

#### Return modes

This op traverses the ops nested under `target` and returns the handles to
all the operations that match the requirements.

This op fails if the target is not a handle to exactly one operation.
Otherwise it succeeds.

This operation does not consume the target handle and produces new handles:
it is a navigation op.
"""
function structured_match(target; results::MLIRType, ops=nothing, interface=nothing, op_attrs=nothing, filter_result_type=nothing, filter_operand_types=nothing, location=Location())
    results = MLIRType[results, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (ops != nothing) && push!(attributes, namedattribute("ops", ops))
    (interface != nothing) && push!(attributes, namedattribute("interface", interface))
    (op_attrs != nothing) && push!(attributes, namedattribute("op_attrs", op_attrs))
    (filter_result_type != nothing) && push!(attributes, namedattribute("filter_result_type", filter_result_type))
    (filter_operand_types != nothing) && push!(attributes, namedattribute("filter_operand_types", filter_operand_types))
    
    create_operation(
        "transform.structured.match", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_multitile_sizes`

Emits the IR computing the tile sizes `s1` and `s2` such that:

  - there exists a combination of `n` tiles of size `s1` and `m` tiles of
    size `s2` that covers the entirety of the iteration space `dimension` of
    the target structured op;
  - `s1`, `s2` is less than or equal to `target_size`;
  - `s1` and `s2` are divisible by `divisor.

For example, for a dimension of size 54 with target size 12 and divisor 2,
this can emit the IR computing the tile size 10, used for 3 tiles, and 12,
used for 2 tiles, totally 10*3 + 12*2 = 54. Note that when the divisor does
not divide the original dimension size, it is impossible to compute such
tile sizes. An assertion is emitted to guard against this in the dynamic
case.

Expects the target size and the divisor to be strictly positive. Folds the
IR as much as possible, normally obtaining constant sizes and numbers of
tiles for a statically known dimension.

This does *not* consume the target handle and produces three handles each
pointing to single-result index-typed operations (which may be arithmetic
constant operations) defining the two respective tile sizes and the product
of the first tile size with the number of tiles of that size (useful for
splitting the iteration space).

This operation composes with the regular tiling when applied per-dimension:

```mlir
%sz1, %sz2, %split = structured.multitile_sizes %target
                     { target_size = 10, dimension = 1 }
                   : !transform.any_op, !transform.param<i64>,
                     !transform.param<i64>, !transform.param<i64>
%low, %high = structured.split %target after %split { dimension = 1 }
            : !transform.any_op, !transform.param<i64>
%tiled_low, %loop1 = structured.tile_using_for %low [0, %sz1]
                   : (!transform.any_op, !transform.param<i64>)
                  -> (!transform.any_op, !transform.any_op)
%tiled_high, %loop2 = structured.tile_using_for %high [0, %sz2]
                    : (!transform.any_op, !transform.param<i64>)
                   -> (!transform.any_op, !transform.any_op)
%common = merge_handles %tiled_low, %tiled_high : !transform.any_op

%sz3, %sz4, %split = structured.multitile_size %target
                     { target_size = 42, dimension = 0 }
                   : !transform.any_op, !transform.any_op,
                     !transform.any_op, !transform.any_op
%sz3r, %sz4r, %splitr = replicate num(%common) %sz3, %sz4, %splitr
         : !transform.any_op, !transform.any_op, !transform.any_op
structured.split %common after %splitr { dimension = 0 }
         : !transform.any_op, !transform.any_op
// ...
```
"""
function structured_multitile_sizes(target; low_size::MLIRType, high_size::MLIRType, split_point::MLIRType, dimension, target_size, divisor=nothing, location=Location())
    results = MLIRType[low_size, high_size, split_point, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension), namedattribute("target_size", target_size), ]
    (divisor != nothing) && push!(attributes, namedattribute("divisor", divisor))
    
    create_operation(
        "transform.structured.multitile_sizes", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_pack_greedily`

Target a Linalg op and rewrite it into packed LinalgOp form by trying to
infer whether a known suboperation is embedded

Different packing strategies are applied in order, when one applies
successfully, the transform returns:
  1. Matmul packing: Try to infer a matmul operation embedded in the target op.
     Specifically, this looks for 2 parallel dimensions that participate in
     an outer-product and 1 reduction dimension.
     These dimensions are referred as (m, n, k) to match canonical matmul
     terminology.

     The packed sizes for (m, n, k) are specified by `matmul_packed_sizes`
     and the optional `matmul_padded_sizes_next_multiple_of`.
     When an entry `matmul_packed_sizes[i]` is non-0, the corresponding
     dimension is packed by `matmul_packed_sizes[i]`.
     Otherwise, the dimension is merely padded to the next multiple of
     `matmul_padded_sizes_next_multiple_of[i]`.

     `matmul_padded_sizes_next_multiple_of` is optional and is expected to
     either be empty or of size `3`, matching the size of `matmul_packed_sizes`.
     For each individual element of `matmul_packed_sizes` and
     `matmul_padded_sizes_next_multiple_of`, only one of them is allowed to
     be non-zero.

     The ordering of the packed dimensions (mm, nn, kk) is specified by the
     `matmul_inner_dims_order` attribute.

Packing occurs as follows:
  1. Find the dimensions to pack according to the strategy.
  2. The target is converted to linalg.generic form.
  3. An interchange transform is applied to isolate the dimensions to pack as
     the most minor indexing dimensions of the linalg.generic. The most minor
     dimensions are themselves ordered according to `inner_dims_order`.
  4. An elementwise traversal of `matmul_packed_sizes` and
     `matmul_padded_sizes_next_multiple_of` is performed and for each
     dimension `d`, either pack to `matmul_packed_sizes[d]` or pad to the
     `matmul_padded_sizes_next_multiple_of[d]`.
  5. Packing/padding is performed by the amounts determined in step 4. and
     following `inner_dims_order`.

By normalizing the most minor dimensions to `inner_dims_order`, the transform
guarantees that packing immediately generates inner dimensions in a desirable
layout.

Outer dimension layout permutations are not controlled by this transform op
at the moment and can be obtained by composing with the pack_transpose
transformation.

#### Return modes

This operation ignores non-Linalg ops and drops them in the return.
It returns the list of packed Linalg ops or the original op when all available
packing strategies failed to apply.
"""
function structured_pack_greedily(target, matmul_packed_sizes; packed_op::MLIRType, static_matmul_packed_sizes=nothing, matmul_padded_sizes_next_multiple_of=nothing, matmul_inner_dims_order=nothing, location=Location())
    results = MLIRType[packed_op, ]
    operands = API.MlirValue[get_value(target), get_value.(matmul_packed_sizes)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (static_matmul_packed_sizes != nothing) && push!(attributes, namedattribute("static_matmul_packed_sizes", static_matmul_packed_sizes))
    (matmul_padded_sizes_next_multiple_of != nothing) && push!(attributes, namedattribute("matmul_padded_sizes_next_multiple_of", matmul_padded_sizes_next_multiple_of))
    (matmul_inner_dims_order != nothing) && push!(attributes, namedattribute("matmul_inner_dims_order", matmul_inner_dims_order))
    
    create_operation(
        "transform.structured.pack_greedily", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_pack`

Pack a LinalgOp by applying a data tiling transformation on the op and
packing the operands according to the `packed_sizes` specification.

Iterator dimensions are tiled in their canonical order in the op spec.
Operands are packed according to the same canonical order of the op iterator
dimensions.

Specifying a packed size of 0 for an iterator removes it from consideration
for packing.

`tensor.pack` (resp. `tensor.unpack`) operations are inserted for the operands
(resp. results) that need to be packed (resp. unpacked) according to the
`packed_sizes` specification.

#### Example

Consider a `linalg.matmul` with indexing maps:
```
  //              M   N   K       M   K
  // affine_map<(d0, d1, d2) -> (d0, d2)>
  //                              K   N
  // affine_map<(d0, d1, d2) -> (d2, d1)>
  //                              M   N
  // affine_map<(d0, d1, d2) -> (d0, d1)>
  %0 = linalg.matmul  ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(    %C: tensor<?x?xf32>)
```

Specifying packed_sizes [2, 3, 4] results in tiling the iterator dimensions
M, N and K, in this order, in both the op and its operands.
```
  //              M   N   K   m   n   k       M   K   m   k
  // affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
  //                                          K   N   n   k
  // affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d4, d5)>
  //                                          M   N   m   n
  // affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
  %0 = linalg.generic_representing_some_higher_d_matmul
        ins(%A, %B: tensor<?x?x2x4xf32>, tensor<?x?x4x3xf32>)
       outs(    %C: tensor<?x?x2x3xf32>)
```
In particular, note that the second operand `B` has shape `KxNxnxk` (and not
`KxNxkxn` as one could expect by looking **only** at the operand).

Other layouts can be obtained unsurprisingly from this canonical
transformation by composing the resulting operation with a
`transform.structured.pack_transpose` op.
This composition allows separating concerns and composes better compared
to adding additional permutation attributes to this transform op.

#### Return modes

This operation applies to a single Linalg op, otherwise it fails.
This operation may produce a definiteFailure if the packing fails for any
reason.

The returned handle point to the packed LinalgOp.
"""
function structured_pack(target, packed_sizes; packed_op::MLIRType, static_packed_sizes=nothing, location=Location())
    results = MLIRType[packed_op, ]
    operands = API.MlirValue[get_value(target), get_value.(packed_sizes)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (static_packed_sizes != nothing) && push!(attributes, namedattribute("static_packed_sizes", static_packed_sizes))
    
    create_operation(
        "transform.structured.pack", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_pack_transpose`

Apply a transposition to a single `tensor.pack` (resp. `tensor.unpack`) and
update the `linalg.generic` op that consumes (resp. produces) the operation.

This transform allows composing a simple `structured.pack` with additional
transpositions to e.g. match the data format required by a specific library
call or ISA instruction.

The transpose spec must specify at least one of `outer_perm` or `inner_perm`
attributes, which will act upon the `outer_dims_perm` or `inner_dims_pos` of
the specified `tensor.pack` or `tensor.unpack` op.

If the `target` of this op is a `tensor.pack` then a new `tensor.empty` will
be created along with transposed versions of the `tensor.pack` and the
consuming `linalg.generic`, which is expected to be the sole consumer.

If the `target` of this op is a `tensor.unpack` then the whole pack / compute
/ unpack chain will be transposed and transposed clones of `tensor.pack`,
the consuming `linalg.generic` and the tail `tensor.pack` will be created.

#### Return modes

This operation targets a single `tensor.pack` / `tensor.unpack` op and a
single matching `linalg.generic` that consumes / produces the op. Otherwise,
it produces a silenceableFailure.

This operation may produce a silenceableFailure if the transpose spec is
ill-formed (i.e. `outer_perm` or `inner_perm` are not permutations of the
proper rank) or if the tranposition of all involved operations fails for any
reason.

This operation returns 3 handles, one to the transformed LinalgOp, one to
the transformed `tensor.pack` and one to the transformed `tensor.unpack`.
The last handle for `tensor.unpack` is empty if `target_pack_or_unpack_op`
was not itself a `tensor.unpack`.
"""
function structured_pack_transpose(target_pack_or_un_pack_op, target_linalg_op; packed_op::MLIRType, pack_op::MLIRType, un_pack_op::MLIRType, outer_perm=nothing, inner_perm=nothing, location=Location())
    results = MLIRType[packed_op, pack_op, un_pack_op, ]
    operands = API.MlirValue[get_value(target_pack_or_un_pack_op), get_value(target_linalg_op), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (outer_perm != nothing) && push!(attributes, namedattribute("outer_perm", outer_perm))
    (inner_perm != nothing) && push!(attributes, namedattribute("inner_perm", inner_perm))
    
    create_operation(
        "transform.structured.pack_transpose", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_pad`

Pads the operations pointed to by the target handle using the options
provides as operation attributes. The operation returns a handle to the
padded operation and to the padding operation (\"tensor.pad\").

To preserve tensor SSA use-def chains, the unpadded result is copied back to
the original destination tensor of the targeted op. The op that copies back
the result can be customized with `copy_back_op`:

* \"bufferization.materialize_in_destination\" (default)
* \"linalg.copy\"
* \"none\" (no copy back)

#### Return modes

This operation ignores non-Linalg ops and drops them in the return.
This operation may produce a definiteFailure if the padding fails for any
reason.

If all the operations referred to by the `target` handle pad
properly, the transform succeeds. Otherwise the transform silently fails.
The return handle points to only the subset of successfully produced
padded operations, which can be empty.
"""
function structured_pad(target; padded::MLIRType, pad::MLIRType, copy::MLIRType, padding_values=nothing, padding_dimensions=nothing, pad_to_multiple_of=nothing, pack_paddings=nothing, transpose_paddings=nothing, copy_back_op=nothing, location=Location())
    results = MLIRType[padded, pad, copy, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (padding_values != nothing) && push!(attributes, namedattribute("padding_values", padding_values))
    (padding_dimensions != nothing) && push!(attributes, namedattribute("padding_dimensions", padding_dimensions))
    (pad_to_multiple_of != nothing) && push!(attributes, namedattribute("pad_to_multiple_of", pad_to_multiple_of))
    (pack_paddings != nothing) && push!(attributes, namedattribute("pack_paddings", pack_paddings))
    (transpose_paddings != nothing) && push!(attributes, namedattribute("transpose_paddings", transpose_paddings))
    (copy_back_op != nothing) && push!(attributes, namedattribute("copy_back_op", copy_back_op))
    
    create_operation(
        "transform.structured.pad", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_promote`

Promotes the specified operands of the target into a separate memory buffer.

At this point, this transform does not allow customizing alloc/dealloc
functions nor the behavior on copy in/out operations.

#### Return modes

This operation applies to a single Linalg op that satisfies the
`promoteSubviewsPrecondition`, otherwise it fails.

If the operations referred to by the `target` handle promote
properly, the transform succeeds.

When successful, the return handle points to the \$target operation that
was modified inplace.
"""
function structured_promote(target; transformed::MLIRType, operands_to_promote=nothing, use_full_tile_buffers=nothing, use_full_tiles_by_default=nothing, use_alloca=nothing, memory_space=nothing, mapping=nothing, alignment=nothing, location=Location())
    results = MLIRType[transformed, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (operands_to_promote != nothing) && push!(attributes, namedattribute("operands_to_promote", operands_to_promote))
    (use_full_tile_buffers != nothing) && push!(attributes, namedattribute("use_full_tile_buffers", use_full_tile_buffers))
    (use_full_tiles_by_default != nothing) && push!(attributes, namedattribute("use_full_tiles_by_default", use_full_tiles_by_default))
    (use_alloca != nothing) && push!(attributes, namedattribute("use_alloca", use_alloca))
    (memory_space != nothing) && push!(attributes, namedattribute("memory_space", memory_space))
    (mapping != nothing) && push!(attributes, namedattribute("mapping", mapping))
    (alignment != nothing) && push!(attributes, namedattribute("alignment", alignment))
    
    create_operation(
        "transform.structured.promote", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_replace`

Replace all `target` payload ops with the single op that is contained in
this op\'s region. All targets must have zero arguments and must be isolated
from above.

This op is for debugging/experiments only.

#### Return modes

This operation consumes the `target` handle.
"""
function structured_replace(target; replacement::MLIRType, bodyRegion::Region, location=Location())
    results = MLIRType[replacement, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[bodyRegion, ]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.structured.replace", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_rewrite_in_destination_passing_style`

Rewrite a supported tensor operation that is not in destination-passing style
into a form that is in destination-passing style.
Currently supported operations are:
  - tensor.pad
  - tensor.generate
  - tensor.from_elements
This dichotomy hints at a future interface, for now the implementation just
switches between different implementation.

#### Return modes

This operation ignores non-unsupported ops and drops them from the return.
If all the operations referred to by the `target` handle generalize
properly, the transform succeeds. Otherwise the transform silently fails.
The return handle points to a subset of successfully produced operations:
  - `tensor.pad` case, the returned handle points to the tensor.insert_slice.
  - `tensor.generate` case, the returned handle points to the linalg.generic.
  - `tensor.from_elements` case, the returned handle points to the last
    `tensor.insert`.
"""
function structured_rewrite_in_destination_passing_style(target; transformed::MLIRType, location=Location())
    results = MLIRType[transformed, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.structured.rewrite_in_destination_passing_style", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_scalarize`

Indicates that ops of a specific kind in the given function should be
scalarized (i.e. their dynamic dimensions tiled by 1).

#### Return modes:

This operation ignores non-Linalg ops and drops them in the return.
This operation produces `definiteFailure` if the scalarization fails for any
reason.
If all the operations referred to by the `target` handle scalarize
properly, the transform succeeds. Otherwise the transform silently fails.

The return handle points to only the subset of successfully produced
tiled-by-1 operations, which can be empty.

This operation does not return handles to the tiled loop.
We make this design choice because it is hard to know ahead of time the
number of loops that will be produced (it depends on the number of dynamic
dimensions after multiple transformations have been applied).
Loops can always be recovered by navigating from the tiled operations if
needed.
"""
function structured_scalarize(target; result::MLIRType, location=Location())
    results = MLIRType[result, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.structured.scalarize", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_specialize`

Transforms a generic operation into the equivalent named form.

#### Return modes

This operation ignores non-Linalg ops and drops them in the return. If all
the operations referred to by the `target` handle specialize, the transform
succeeds; otherwise, the operation produces a silenceable failure.  The return
handle points to only the subset of successfully produced equivalent named
operations, which can be empty or contain the original ops if they were already
in named form. The supported specialization to named Linalg operations are:
- linalg.copy of any rank.
"""
function structured_specialize(target; transformed::MLIRType, location=Location())
    results = MLIRType[transformed, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.structured.specialize", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_split`

Indicates that the given `target` op should be split into two complementary
parts, which combined cover the entire iteration domain of the original op.
The split is performed along the iteration space dimension provided as
attribute. In case of dimension overflow, the transformation fails. The
split is performed at the dimension iterator value specified as either the
static split point attribute when it is known at transform IR construction
time or as the handle to an operation producing a single index-typed value
when it is computed by payload IR. In the latter case, the static split
point must be set to `ShapedType::kDynamic` and the dynamic size handle
must point to as many value-producing operations as there are structured
operations pointed to by the target handle.

The operation consumes the target handle, but preserves the split point
handle if provided. It produces two new handles pointing to the two parts
of the structured op after splitting, in the same order as the target
operand, with the first handle corresponding to the part with lower
iteration space indices.
"""
function structured_split(target, dynamic_split_point=nothing; first::MLIRType, second::MLIRType, dimension, static_split_point, location=Location())
    results = MLIRType[first, second, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension), namedattribute("static_split_point", static_split_point), ]
    (dynamic_split_point != nothing) && push!(operands, get_value(dynamic_split_point))
    
    create_operation(
        "transform.structured.split", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_split_reduction`

Indicates that the given `target` op should be transformed with the
`splitReduction` transformation and split factor provided as attribute.

The `splitReduction` transformation splits the first single linalg op
reduction into a parallel and reduction dimension.
A new `linalg.generic` op is created to perform the rest of the reduction.

The transformation supports different configurations attributes:
  - split_factor: the factor by which to split (i.e. the size of the
    remaining reduction after splitting).
  - insert_split_dimension: the dimension in the temporary tensor into
    which the new parallel dimension is inserted.
  - inner_parallel: specifies whether the parallel dimension is before or
    after the reduction dimension in the splitting op.
  - use_scaling_algorithm: whether to use a scaling based formulation that
    does not create an ExpandShapeOp (default: do not use scaling)
  - use_alloc: whether to use an alloc op to allocate the temporary
    tensor (default: do not use alloc op)

#### Return modes

This operation ignores non-Linalg ops and drops them in the return.
This operation produces `definiteFailure` if the splitting fails for any
reason.

If all the operations referred to by the `target` handle split
properly, the transform succeeds. Otherwise the transform silently fails.
The 4 returned handles points to only the subset of successfully produced
computational operations, which can all be empty.
This 4 returned handles point to:
  - the init op (or tensor_alloc op if use_alloc = true),
  - the fill op used to initialize the neutral element,
  - the split op and
  - the result-combining op.

#### Example (default: `use_scaling_algorithm = false, use_alloc = false`):

```
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                        affine_map<(d0) -> ()>],
        iterator_types = [\"reduction\"]}
  ins(%in : tensor<32xf32>)
  outs(%out : tensor<f32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %y = arith.addf %arg1, %arg2 : f32
    linalg.yield %y : f32
  } -> tensor<f32>
```

is split into:

```
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.expand_shape %in [[0, 1]] : tensor<32xf32> into tensor<4x8xf32>
  %1 = tensor.empty() : tensor<4xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<4xf32>) -> tensor<4xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                        affine_map<(d0, d1) -> (d0)>],
    iterator_types = [\"parallel\", \"reduction\"]}
    ins(%0 : tensor<4x8xf32>) outs(%2 : tensor<4xf32>) {
    ^bb0(%arg3: f32, %arg5: f32):
    %5 = arith.addf %arg3, %arg4 : f32
    linalg.yield %5 : f32
  } -> tensor<4xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                        affine_map<(d0) -> ()>],
    iterator_types = [\"reduction\"]}
    ins(%3 : tensor<4xf32>) outs(%out : tensor<f32>) {
    ^bb0(%arg3: f32, %arg4: f32):
    %5 = arith.addf %arg3, %arg4 : f32
    linalg.yield %5 : f32
  } -> tensor<f32>
```

#### Example (`use_scaling_algorithm = true, use_alloc = true`):

Instead of introducing an ExpandShapeOp, this scaling-based implementation
rewrites a reduction dimension `k` into `k * split_factor + kk`.
The dimension `kk` is added as an extra parallel dimension to the
intermediate output tensor at position `insert_split_dimension`.

Consider a minimal example where `k` is reduced:
    O(i, j) += I(i, j, k)
Assume i=3, j=5, k=128, split_factor=16 and insert_split_dimension=0.
The compute is rewritten as:
  a. O_i(kk, i, j) += I(i, j, 16 * k + kk)
  b. O(i, j) += O_i(kk, i, j)
The intermediate tensor O_i is of shape (128/16)x3x5 == 8x3x5.

#### Example:

```
 %0 = linalg.matmul ins(%A, %B: tensor<16x256xf32>, tensor<256x32xf32>)
   outs(%C: tensor<16x32xf32>) -> tensor<16x32xf32>
```

Is transformed to:

```
 #map0 = affine_map<(d0, d1, d2, d3) -> (d0, d2 * 4 + d3)>
 #map1 = affine_map<(d0, d1, d2, d3) -> (d2 * 4 + d3, d1)>
 #map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
 #map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
 #map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
 #map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
 %0 = tensor.empty() : tensor<16x32x64xf32>
 %cst = arith.constant 0.000000e+00 : f32
 %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x32x64xf32>) ->
    tensor<16x32x64xf32>
 %2 = tensor.empty() : tensor<64x4xi1>

 %3 = linalg.generic {indexing_maps = [#map0, #map1, #map2, #map3],
   iterator_types = [\"parallel\", \"parallel\", \"parallel\", \"reduction\"]}
   ins(%A, %B, %2 : tensor<16x256xf32>, tensor<256x32xf32>, tensor<64x4xi1>)
   outs(%1 : tensor<16x32x64xf32>) {
     ^bb0(%arg3: f32, %arg4: f32, %arg5: i1, %arg6: f32):
       %5 = arith.mulf %arg3, %arg4 : f32
       %6 = arith.addf %arg6, %5 : f32
       linalg.yield %6 : f32
 } -> tensor<16x32x64xf32>

 %4 = linalg.generic {indexing_maps = [#map4, #map5],
   iterator_types = [\"parallel\", \"parallel\", \"reduction\"]}
   ins(%3 : tensor<16x32x64xf32>)
   outs(%C : tensor<16x32xf32>) {
     ^bb0(%arg3: f32, %arg4: f32):
       %5 = arith.addf %arg3, %arg4 : f32
       linalg.yield %5 : f32
 } -> tensor<16x32xf32>

 return %4 : tensor<16x32xf32>
```
"""
function structured_split_reduction(target; init_or_alloc_op::MLIRType, fill_op::MLIRType, split_linalg_op::MLIRType, combining_linalg_op::MLIRType, split_factor=nothing, insert_split_dimension=nothing, inner_parallel=nothing, use_scaling_algorithm=nothing, use_alloc=nothing, location=Location())
    results = MLIRType[init_or_alloc_op, fill_op, split_linalg_op, combining_linalg_op, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (split_factor != nothing) && push!(attributes, namedattribute("split_factor", split_factor))
    (insert_split_dimension != nothing) && push!(attributes, namedattribute("insert_split_dimension", insert_split_dimension))
    (inner_parallel != nothing) && push!(attributes, namedattribute("inner_parallel", inner_parallel))
    (use_scaling_algorithm != nothing) && push!(attributes, namedattribute("use_scaling_algorithm", use_scaling_algorithm))
    (use_alloc != nothing) && push!(attributes, namedattribute("use_alloc", use_alloc))
    
    create_operation(
        "transform.structured.split_reduction", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_tile_reduction_using_for`

Indicates that the given `target` op should be transformed with the
`tileReduction` transformation with the tile size provided as attribute.

This transformation tiles the `target` along the reduction dimensions. It
creates a tensor initialized with the identity value. Then it creates nested
loops with a parallel version of `target` op inside. The parallel op
dimensions are less or equal to the tile size passed by user.
After the loop a merge operation is created to do a final reduction with the
partial reductions.
The initial tensor always uses the tile size dimension. This may overallocate
if the tile size is greater than the reduction dimension.

#### Return modes

Returns 4 handles associated with (in order):
  - the fill op used to initialize the neutral element,
  - the parallel tiled op and
  - the result-combining op,
  - the parent `for` op.

#### Example:

```
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
  iterator_types = [\"parallel\", \"reduction\"]}
  ins(%arg0 : tensor<?x?xf32>)
  outs(%out : tensor<?xf32>) {
    ^bb0(%arg7: f32, %arg9: f32):
    %1 = arith.addf %arg7, %arg9 : f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %red : tensor<?xf32>
```

is transformed into:

```
  %0 = tensor.empty(%dim_1) : tensor<?x5xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x5xf32>) -> tensor<?x5xf32>
  %2 = scf.for %arg2 = %c0 to %dim_0 step %c5 iter_args(%arg3 = %1) -> (tensor<?x5xf32>) {
    %extracted_slice = tensor.extract_slice %1[0, 0] [%dim, 5] [1, 1] : tensor<?x5xf32> to tensor<?x5xf32>
    %extracted_slice_2 = tensor.extract_slice %arg0[0, %arg2] [%dim, 5] [1, 1] : tensor<?x?xf32> to tensor<?x5xf32>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [\"parallel\", \"parallel\"]}
    ins(%extracted_slice_2 : tensor<?x5xf32>)
    outs(%extracted_slice : tensor<?x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %in, %out : f32
      linalg.yield %5 : f32
    } -> tensor<?x5xf32>
    %dim_3 = tensor.dim %1, %c0 : tensor<?x5xf32>
    %inserted_slice = tensor.insert_slice %4 into %arg3[0, 0] [%dim_3, 5] [1, 1] : tensor<?x5xf32> into tensor<?x5xf32>
    scf.yield %inserted_slice : tensor<?x5xf32>
  }
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                        affine_map<(d0, d1) -> (d0)>],
  iterator_types = [\"parallel\", \"reduction\"]}
  ins(%2 : tensor<?x5xf32>)
  outs(%arg1 : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %4 = arith.addf %in, %out : f32
    linalg.yield %4 : f32
  } -> tensor<?xf32>
```
"""
function structured_tile_reduction_using_for(target; fill_op::MLIRType, split_linalg_op::MLIRType, combining_linalg_op::MLIRType, for_op::MLIRType, tile_sizes=nothing, location=Location())
    results = MLIRType[fill_op, split_linalg_op, combining_linalg_op, for_op, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (tile_sizes != nothing) && push!(attributes, namedattribute("tile_sizes", tile_sizes))
    
    create_operation(
        "transform.structured.tile_reduction_using_for", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_tile_reduction_using_forall`

Tile a PartialReductionOpInterface op to a tiled `scf.forall` doing
partial reduction.

This transformation tiles the `target` along the reduction dimensions. It
creates a tensor initialized with the identity value. Then it creates a
`scf.forall` loops with the number threads given by `num_threads`.
The op is tiled op with a size equal to `floordiv(size, num_threads)`.
All the partial reduction value is are parallel inserted to create a new
tensor. After the loop a merge operation is created to do a final reduction
with the partial reductions tensor.
If an extra `tile_sizes` parameter is passed the tiles are cyclically
distributed on the threads of the `scf.foralls` loop.

#### Return modes

Returns 4 handles associated with (in order):
  - the fill op used to initialize the neutral element,
  - the parallel tiled op and
  - the result-combining op,
  - the parent `forall` op.

#### Example:

```
  %red = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0)>],
  iterator_types = [\"parallel\", \"reduction\"]}
  ins(%arg0 : tensor<?x?xf32>)
  outs(%out : tensor<?xf32>) {
    ^bb0(%arg7: f32, %arg9: f32):
    %1 = arith.addf %arg7, %arg9 : f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %red : tensor<?xf32>
```

is transformed into:

```
  %0 = tensor.empty(%dim_1) : tensor<?x5xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x5xf32>) -> tensor<?x5xf32>
  %2 = scf.forall (%arg2) in (%c5) shared_outs(%arg3 = %1) -> (tensor<?x5xf32>) {
    %4 = affine.min #map(%arg2)[%dim_0]
    %5 = affine.max #map1(%4)
    %extracted_slice = tensor.extract_slice %arg3[0, %arg2] [%dim, 1] [1, 1] : tensor<?x5xf32> to tensor<?xf32>
    %6 = affine.apply #map2(%arg2)[%dim_0]
    %extracted_slice_2 = tensor.extract_slice %arg0[0, %6] [%dim, %5] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
    %extracted_slice_3 = tensor.extract_slice %extracted_slice[0] [%dim] [1] : tensor<?xf32> to tensor<?xf32>
    %7 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = [\"parallel\", \"reduction\"]} ins(%extracted_slice_2 : tensor<?x?xf32>) outs(%extracted_slice_3 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.addf %in, %out : f32
      linalg.yield %9 : f32
    } -> tensor<?xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %7 into %arg3[0, %arg2] [%dim, 1] [1, 1] : tensor<?xf32> into tensor<?x5xf32>
    }
  } {mapping = []}
  %3 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = [\"parallel\", \"reduction\"]} ins(%2 : tensor<?x5xf32>) outs(%arg1 : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %4 = arith.addf %in, %out : f32
    linalg.yield %4 : f32
  } -> tensor<?xf32>
```
"""
function structured_tile_reduction_using_forall(target; fill_op::MLIRType, split_linalg_op::MLIRType, combining_linalg_op::MLIRType, forall_op::MLIRType, num_threads=nothing, tile_sizes=nothing, mapping=nothing, location=Location())
    results = MLIRType[fill_op, split_linalg_op, combining_linalg_op, forall_op, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (num_threads != nothing) && push!(attributes, namedattribute("num_threads", num_threads))
    (tile_sizes != nothing) && push!(attributes, namedattribute("tile_sizes", tile_sizes))
    (mapping != nothing) && push!(attributes, namedattribute("mapping", mapping))
    
    create_operation(
        "transform.structured.tile_reduction_using_forall", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_tile_using_for`

Indicates that the given `target` op should be tiled with the given sizes.
This transform generates a loop nest with a smaller (\"tiled\") target
operation in its body. Currently limited to LinalgOps.

Tile sizes may be known at transformation time, in which case they are
expected to be provided in the `static_size` attribute, or not, in which
case the tile value must be computed by the payload IR and the handle to the
operation computing it must be provided through `dynamic_sizes`. When the
sizes are not known statically, the corresponding entry in the
`static_sizes` attribute must be set to `ShapedType::kDynamic`. Only
the dynamic sizes must be provided in `dynamic_sizes`, i.e., there should
be as many handles as `ShapedType::kDynamic` values in the
`static_sizes` attribute. A static size of `0` indicates that the dimension
should not be tiled. No loop will be generated for such dimensions. If all
tile sizes are `0`, this transform is effectively a no-op.

This op returns handles to the tiled op (in the generated loop nest) and the
generated loops. The number of loops is the number of tile sizes that are
statically known to be non-zero.

#### Return modes

On success, the resulting handles are associated with co-indexed lists of
tiled operations and loops around them.

This operation only supports Linalg ops and produces a silenceable failure
if the input contains any non-Linalg ops. The ops preceding it in the list
associated with the `target` handle will have been tiled.

This operation produces a silenceable failure if the `dynamic_sizes` handles
are associated with lists of payload operations of a size different than
that of the list associated with the `target` handle.

If the internal implementation of tiling for any of the operations fails,
produces a definite failure.
"""
function structured_tile_using_for(target, dynamic_sizes; tiled_linalg_op::MLIRType, loops::Vector{MLIRType}, static_sizes=nothing, interchange=nothing, scalable_sizes=nothing, location=Location())
    results = MLIRType[tiled_linalg_op, loops..., ]
    operands = API.MlirValue[get_value(target), get_value.(dynamic_sizes)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (static_sizes != nothing) && push!(attributes, namedattribute("static_sizes", static_sizes))
    (interchange != nothing) && push!(attributes, namedattribute("interchange", interchange))
    (scalable_sizes != nothing) && push!(attributes, namedattribute("scalable_sizes", scalable_sizes))
    
    create_operation(
        "transform.structured.tile_using_for", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_tile_using_forall`

Tile a TilingInterface op to a tiled `scf.forall`.

Tiling is applied by either specifying `num_threads` or `tile_size`. If
`num_threads` is specified, then the tile size for each dimension `i` is
calculated dynamically via `ceilDiv(dimSize[i], num_threads[i])`.
`num_threads` and `tile_size` can be either static index attributes or 
operation handles (or a mix thereof). Operation handles must be mapped to
exactly one op that has exactly one result of index type.

Static zero tile sizes indicate that the dimension is not tiled and can be
thought of as tiling by the full size of data.

It is the user\'s responsibility to ensure that `num_threads/tile_sizes` is
a valid tiling specification (i.e. that only tiles parallel dimensions,
e.g. in the Linalg case).

If non-empty, the `mapping` is added as an attribute to the
resulting `scf.forall`.

Note: `tile_sizes` and `num_threads` are variadic. Each tile size/number of
threads can be an index attribute or a transform handle that is mapped to
exactly one payload op with exactly one index result.

#### Return modes

This operation ignores ops that do not implement the TilingInterface and
drops them in the return.

If all the operations referred to by the `target` handle tile
successfully, the transform succeeds.
Otherwise the transform silently fails.

The two returned handles point to only the subset of successfully produced
tiled operations, which can all be empty.

These two returned handles point to:
  - the tiled op that implements TilingInterface,
  - the new scf.forall op.

#### Example using `num_threads`

```
%0 = transform.structured.match ops{[\"linalg.matmul\"]} in %arg1
   : (!transform.any_op) -> !transform.any_op
%3:2 = transform.structured.tile_using_forall %0 num_threads [10, 20]
   : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
```

#### Example using `tile_sizes`

```
%0 = transform.structured.match ops{[\"linalg.matmul\"]} in %arg1
   : (!transform.any_op) -> !transform.any_op
%sz = transform.structured.match ...
%3:2 = transform.structured.tile_using_forall %0 tile_sizes [0, %sz, 20]
   : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
```
"""
function structured_tile_using_forall(target, num_threads, tile_sizes, packed_num_threads=nothing; packed_tile_sizes=nothing, tiled_op::MLIRType, forall_op::MLIRType, static_num_threads=nothing, static_tile_sizes=nothing, mapping=nothing, location=Location())
    results = MLIRType[tiled_op, forall_op, ]
    operands = API.MlirValue[get_value(target), get_value.(num_threads)..., get_value.(tile_sizes)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (packed_num_threads != nothing) && push!(operands, get_value(packed_num_threads))
    (packed_tile_sizes != nothing) && push!(operands, get_value(packed_tile_sizes))
    push!(attributes, operandsegmentsizes([1, length(num_threads), length(tile_sizes), (packed_num_threads==nothing) ? 0 : 1(packed_tile_sizes==nothing) ? 0 : 1]))
    (static_num_threads != nothing) && push!(attributes, namedattribute("static_num_threads", static_num_threads))
    (static_tile_sizes != nothing) && push!(attributes, namedattribute("static_tile_sizes", static_tile_sizes))
    (mapping != nothing) && push!(attributes, namedattribute("mapping", mapping))
    
    create_operation(
        "transform.structured.tile_using_forall", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_transpose_conv2d`

Convert linalg.conv_2d_nhwc_fhwc into linalg.conv_2d_nhwc_hwcf by introducing
a linalg.transpose on the filter tensor/memref.

Whilst the fhwc filter channel ordering can be desirable for certain targets
and is a more direct mapping to higher level dialects such as TOSA (which only
supports this ordering) hwcf is better suited for transformations such as
img2col which can make use of optimized BLAS routines such as GEMM.

Returns one handle:
- The final operation of the sequence that replaces the original
  convolution.

#### Return modes:

Returns a definite failure if target is not isolated from above.
Returns a silenceable failure if the pattern application failed.
"""
function structured_transpose_conv2d(target; transformed::MLIRType, location=Location())
    results = MLIRType[transformed, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "transform.structured.transpose_conv2d", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_vectorize_children_and_apply_patterns`

Vectorizes all children contained in the given `target` using the
configuration specified by the attributes of this op. This only vectorizes
structured ops that operate on shaped types and does not vectorize loops or
straight-line. Internally, it applies a set of rewrite patterns, some of
which enable vectorization and some of which clean up the results.
Therefore, it can only be applied to an op with the \"isolated from above\"
property. This transformation only fails if the entire pattern rewriting
failed, i.e., it does **not** fail when no ops were vectorized.

Finer granularity can be achieved either with the `VectorizeOp` for
individual ops or by outlining the target part of the payload IR into, e.g.,
a function, performing this transformation, and inlining it back.

Note that this transformation invalidates the handles to any payload IR
operation that is contained inside the vectorization target.

This transformation supports the following attributes:
- `vectorize_padding`: a `UnitAttr` to activate the vectorization of
  `tensor.pad` ops. Different pipelines may prefer to lower such ops to
  loops.
- `disable_multi_reduction_to_contract_patterns`: a `UnitAttr` to deactivate
  the rewrite of `vector.multi_reduction` to `vector.contract`. This is
  intended to be used in tests only.
- `disable_transfer_permutation_map_lowering_patterns`: a `UnitAttr` to
  deactivate the rewrite of `vector.transfer` with permutation maps into
  explicit `vector.transpose` operations. This is intended to be used in
  tests only but may be promoted to a first class attribute in the future.

#### Return modes:

This operation produces `definiteFailure` if vectorization fails for any
reason.
The operation always returns the handle to the target op that is expected
to be isolated from above.
"""
function structured_vectorize_children_and_apply_patterns(target; transformed::MLIRType, vectorize_padding=nothing, vectorize_nd_extract=nothing, flatten_1d_depthwise_conv=nothing, disable_multi_reduction_to_contract_patterns=nothing, disable_transfer_permutation_map_lowering_patterns=nothing, location=Location())
    results = MLIRType[transformed, ]
    operands = API.MlirValue[get_value(target), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (vectorize_padding != nothing) && push!(attributes, namedattribute("vectorize_padding", vectorize_padding))
    (vectorize_nd_extract != nothing) && push!(attributes, namedattribute("vectorize_nd_extract", vectorize_nd_extract))
    (flatten_1d_depthwise_conv != nothing) && push!(attributes, namedattribute("flatten_1d_depthwise_conv", flatten_1d_depthwise_conv))
    (disable_multi_reduction_to_contract_patterns != nothing) && push!(attributes, namedattribute("disable_multi_reduction_to_contract_patterns", disable_multi_reduction_to_contract_patterns))
    (disable_transfer_permutation_map_lowering_patterns != nothing) && push!(attributes, namedattribute("disable_transfer_permutation_map_lowering_patterns", disable_transfer_permutation_map_lowering_patterns))
    
    create_operation(
        "transform.structured.vectorize_children_and_apply_patterns", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`structured_vectorize`

Vectorize the target ops, which must be Linalg ops. 

Use the optional vector sizes to specify exactly what configuration the
vectorizer should use. It will then use masked vectors of the specified
size to enforce this configuration (\"masked vectorization\"). If no vector
sizes are specified, the vectorizer will infer the shapes to use from the
target Linalg ops (\"regular vectorization\"). More specifically:

```mlir
# Masked vectorization - vector sizes are specified explicitly
transform.structured.vectorize %target vector_sizes [1, 4] : !transform.any_op
# Regular vectorization - vector sizes are inferred from the target Op
transform.structured.vectorize %target : !transform.any_op
```

The vector sizes can be either static or dynamic (SSA values). In case of
SSA values, the handle must be mapped to exactly one payload op with
exactly one index-typed result.

Note: The input vector sizes must be bigger than or equal to their
counterpart iteration space sizes.

Typically this operator should be applied to linalg operations that have
already been tiled to the appropriate sizes.

#### Return modes:

This operation produces a silenceable failure if at least one target op is
not a Linalg op or fails to vectorize. It produces a definite failure if
the dynamic vector sizes (SSA values) do not satisfy the constraints
mentioned above.
"""
function structured_vectorize(target, vector_sizes; vectorize_nd_extract=nothing, scalable_sizes=nothing, static_vector_sizes=nothing, location=Location())
    results = MLIRType[]
    operands = API.MlirValue[get_value(target), get_value.(vector_sizes)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (vectorize_nd_extract != nothing) && push!(attributes, namedattribute("vectorize_nd_extract", vectorize_nd_extract))
    (scalable_sizes != nothing) && push!(attributes, namedattribute("scalable_sizes", scalable_sizes))
    (static_vector_sizes != nothing) && push!(attributes, namedattribute("static_vector_sizes", static_vector_sizes))
    
    create_operation(
        "transform.structured.vectorize", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end


end # transform
