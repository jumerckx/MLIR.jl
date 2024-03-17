module gpu

import ...IR: IR, NamedAttribute, get_value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`all_reduce`

The `all_reduce` op reduces the value of every work item across a local
workgroup. The result is equal for all work items of a workgroup.

For example, both

```mlir
%1 = gpu.all_reduce add %0 {} : (f32) -> (f32)
%2 = gpu.all_reduce %0 {
^bb(%lhs : f32, %rhs : f32):
  %sum = arith.addf %lhs, %rhs : f32
  \"gpu.yield\"(%sum) : (f32) -> ()
} : (f32) -> (f32)
```

compute the sum of each work item\'s %0 value. The first version specifies
the accumulation as operation, whereas the second version specifies the
accumulation as code region. The reduction operation must be one of:
*  Integer types: `add`, `mul`, `minui`, `minsi`, `maxui`, `maxsi`, `and`,
   `or`, `xor`
*  Floating point types: `add`, `mul`, `minnumf`, `maxnumf`, `minimumf`,
   `maximumf`

If `uniform` flag is set either none or all work items of a workgroup
need to execute this op in convergence.
"""
function all_reduce(value; result=nothing::Union{Nothing, IR.Type}, op=nothing, uniform=nothing, body::Region, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value(value), ]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    !isnothing(op) && push!(attributes, namedattribute("op", op))
    !isnothing(uniform) && push!(attributes, namedattribute("uniform", uniform))
    
    create_operation(
        "gpu.all_reduce", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`alloc`

The `gpu.alloc` operation allocates a region of memory on the GPU. It is
similar to the `memref.alloc` op, but supports asynchronous GPU execution.

The op does not execute before all async dependencies have finished
executing.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it also returns a !gpu.async.token.

If the `host_shared` keyword is present, the memory will be allocated in a
memory accessible both on host and on device.

# Example

```mlir
%memref, %token = gpu.alloc async [%dep] host_shared (%width) : memref<64x?xf32, 1>
```
"""
function alloc(asyncDependencies, dynamicSizes, symbolOperands; memref::IR.Type, asyncToken=nothing::Union{Nothing, IR.Type}, hostShared=nothing, location=Location())
    results = IR.Type[memref, ]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value.(dynamicSizes)..., get_value.(symbolOperands)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(asyncDependencies), length(dynamicSizes), length(symbolOperands), ]))
    !isnothing(asyncToken) && push!(results, asyncToken)
    !isnothing(hostShared) && push!(attributes, namedattribute("hostShared", hostShared))
    
    create_operation(
        "gpu.alloc", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`barrier`

The \"barrier\" op synchronizes all work items of a workgroup. It is used
to coordinate communication between the work items of the workgroup.

```mlir
gpu.barrier
```

waits until all work items in the workgroup have reached this point
and all memory accesses made by these work items prior to the op are
visible to all work items in the workgroup. Data hazards between work items
accessing the same memory can be avoided by synchronizing work items
in-between these accesses.

Either none or all work items of a workgroup need to execute this op
in convergence.
"""
function barrier(; location=Location())
    results = IR.Type[]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "gpu.barrier", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`binary`

GPU binaries provide a semantic mechanism for storing GPU objects,
e.g. the result of compiling a GPU module to an object file.

This operation has 3 arguments:
 - The name of the binary.
 - An optional attribute implementing the offloading LLVM translation interface.
 - An array of GPU object attributes.

During translation, the offloading attribute will be called for translating
GPU `binary` and `launch_func` operations. The default offloading handler is:
`#gpu.select_object`, this handler selects the first object from the array
and embeds it as a string.

Examples:
```
  // Selects the first object.
  gpu.binary @myobject [#gpu.object<...>, #gpu.object<...>]
  // Uses the `#foo.my_handler` for handling the binary during translation.
  gpu.binary @myobject <#foo.my_handler> [#gpu.object<...>, #gpu.object<...>]
  // Selects the object with the `#rocdl.target` target attribute.
  gpu.binary @myobject <#gpu.select_object<#rocdl.target>> [#gpu.object<...>, #gpu.object<#rocdl.target, ...>]
```
"""
function binary(; sym_name, offloadingHandler=nothing, objects, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("sym_name", sym_name), namedattribute("objects", objects), ]
    !isnothing(offloadingHandler) && push!(attributes, namedattribute("offloadingHandler", offloadingHandler))
    
    create_operation(
        "gpu.binary", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`block_dim`

Returns the number of threads in the thread block (aka the block size) along
the x, y, or z `dimension`.

# Example

```mlir
%bDimX = gpu.block_dim x
```
"""
function block_dim(; result_0=nothing::Union{Nothing, IR.Type}, dimension, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension), ]
    !isnothing(result_0) && push!(results, result_0)
    
    create_operation(
        "gpu.block_dim", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`block_id`

Returns the block id, i.e. the index of the current block within the grid
along the x, y, or z `dimension`.

# Example

```mlir
%bIdY = gpu.block_id y
```
"""
function block_id(; result_0=nothing::Union{Nothing, IR.Type}, dimension, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension), ]
    !isnothing(result_0) && push!(results, result_0)
    
    create_operation(
        "gpu.block_id", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`cluster_dim`

Returns the number of thread blocks in the cluster along
the x, y, or z `dimension`.

# Example

```mlir
%cDimX = gpu.cluster_dim x
```
"""
function cluster_dim(; result_0=nothing::Union{Nothing, IR.Type}, dimension, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension), ]
    !isnothing(result_0) && push!(results, result_0)
    
    create_operation(
        "gpu.cluster_dim", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`cluster_id`

Returns the cluster id, i.e. the index of the current cluster within the
grid along the x, y, or z `dimension`.

# Example

```mlir
%cIdY = gpu.cluster_id y
```
"""
function cluster_id(; result_0=nothing::Union{Nothing, IR.Type}, dimension, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension), ]
    !isnothing(result_0) && push!(results, result_0)
    
    create_operation(
        "gpu.cluster_id", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`create_2to4_spmat`

The `gpu.create_2to4_spmat` operation initializes a sparse matrix in dense
format with 2:4 sparsity.
The buffers must already be copied from the host to the device prior to
using this operation. The operation returns a handle to the sparse
matrix descriptor.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token in addition to the environment.

# Example

```mlir
%spmat, %token = gpu.create_2to4_spmat async [%dep] {PRUNE_AND_CHECK} %rows, %cols, %mem: memref<?xf64>
```
"""
function create_2to4_spmat(asyncDependencies, rows, cols, memref; spMat::IR.Type, asyncToken=nothing::Union{Nothing, IR.Type}, pruneFlag=nothing, location=Location())
    results = IR.Type[spMat, ]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(rows), get_value(cols), get_value(memref), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncToken) && push!(results, asyncToken)
    !isnothing(pruneFlag) && push!(attributes, namedattribute("pruneFlag", pruneFlag))
    
    create_operation(
        "gpu.create_2to4_spmat", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`create_bsr`

The `gpu.create_bsr` operation initializes a sparse matrix in BSR format
with the given sizes for the matrix and blocks from the given position,
index, and values buffers. The buffers must already be copied from the
host to the device prior to using this operation. The operation returns
a handle to the sparse matrix descriptor.

The BSR format is similar to CSR, where the column indices represent
two-dimensional blocks instead of a single matrix entry. Note that this
operation (currently) only supports storage with **square** blocks,
i.e., `rBlockSize == cBlockSize`.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token in addition to the environment.

# Example

```mlir
%spmat, %token = gpu.create_bsr async [%dep]
   %brows, %bcols, %bnnz, %rBlockSize, %cBlockSize,
   %bRowPos, %bColIdxs, %values : memref<?xindex>, memref<?xindex>, memref<?xf64>
```
"""
function create_bsr(asyncDependencies, brows, bcols, bnnz, rBlockSize, cBlockSize, bRowPos, bColIdxs, values; spmat::IR.Type, asyncToken=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[spmat, ]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(brows), get_value(bcols), get_value(bnnz), get_value(rBlockSize), get_value(cBlockSize), get_value(bRowPos), get_value(bColIdxs), get_value(values), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncToken) && push!(results, asyncToken)
    
    create_operation(
        "gpu.create_bsr", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`create_coo_aos`

The `gpu.create_coo_aos` operation initializes a sparse matrix in COO format
with the given sizes from the given index and values buffers. The buffers
must already be copied from the host to the device prior to using this
operation. The operation returns a handle to the sparse matrix descriptor.
Unlike the default `gpu.create_coo` operation, this operation builds the
COO format from a single index buffer in AoS format (note that this
feature has been deprecated in cuSparse 11.2).

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token in addition to the environment.

# Example

```mlir
%spmat, %token = gpu.create_coo_aos async [%dep] %rows, %cols, %nnz, %idxs,
    %values : memref<?xindex>, memref<?xf64>
```
"""
function create_coo_aos(asyncDependencies, rows, cols, nnz, idxs, values; spmat::IR.Type, asyncToken=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[spmat, ]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(rows), get_value(cols), get_value(nnz), get_value(idxs), get_value(values), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncToken) && push!(results, asyncToken)
    
    create_operation(
        "gpu.create_coo_aos", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`create_coo`

The `gpu.create_coo` operation initializes a sparse matrix in COO format
with the given sizes from the given index and values buffers. The buffers
must already be copied from the host to the device prior to using this
operation. The operation returns a handle to the sparse matrix descriptor.
Note that this operation builds the COO in SoA format.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token in addition to the environment.

# Example

```mlir
%spmat, %token = gpu.create_coo async [%dep] %rows, %cols, %nnz, %rowIdx,
    %colIdx, %values : memref<?xindex>, memref<?xindex>, memref<?xf64>
```
"""
function create_coo(asyncDependencies, rows, cols, nnz, rowIdxs, colIdxs, values; spmat::IR.Type, asyncToken=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[spmat, ]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(rows), get_value(cols), get_value(nnz), get_value(rowIdxs), get_value(colIdxs), get_value(values), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncToken) && push!(results, asyncToken)
    
    create_operation(
        "gpu.create_coo", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`create_csc`

The `gpu.create_csc` operation initializes a sparse matrix in CSC format
with the given sizes from the given position, index, and values buffers.
The buffers must already be copied from the host to the device prior to
using this operation. The operation returns a handle to the sparse
matrix descriptor.

The CSC format has exactly the same memory layout as its transpose
in CSR format (and vice versa).

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token in addition to the environment.

# Example

```mlir
%spmat, %token = gpu.create_csc async [%dep] %rows, %cols, %nnz, %colPos,
    %rowIdx, %values : memref<?xindex>, memref<?xindex>, memref<?xf64>
```
"""
function create_csc(asyncDependencies, rows, cols, nnz, colPos, rowIdxs, values; spmat::IR.Type, asyncToken=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[spmat, ]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(rows), get_value(cols), get_value(nnz), get_value(colPos), get_value(rowIdxs), get_value(values), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncToken) && push!(results, asyncToken)
    
    create_operation(
        "gpu.create_csc", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`create_csr`

The `gpu.create_csr` operation initializes a sparse matrix in CSR format
with the given sizes from the given position, index, and values buffers.
The buffers must already be copied from the host to the device prior to
using this operation. The operation returns a handle to the sparse
matrix descriptor.

The CSR format has exactly the same memory layout as its transpose
in CSC format (and vice versa).

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token in addition to the environment.

# Example

```mlir
%spmat, %token = gpu.create_csr async [%dep] %rows, %cols, %nnz, %rowPos,
    %colIdx, %values : memref<?xindex>, memref<?xindex>, memref<?xf64>
```
"""
function create_csr(asyncDependencies, rows, cols, nnz, rowPos, colIdxs, values; spmat::IR.Type, asyncToken=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[spmat, ]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(rows), get_value(cols), get_value(nnz), get_value(rowPos), get_value(colIdxs), get_value(values), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncToken) && push!(results, asyncToken)
    
    create_operation(
        "gpu.create_csr", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`create_dn_tensor`

The `gpu.create_dn_tensor` operation initializes a dense tensor from
the given values buffer and sizes. The buffer must already be copied
from the host to the device prior to using this operation. The
operation returns a handle to the dense tensor descriptor.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token in addition to the environment.

# Example

```mlir
%dmat, %token = gpu.create_dn_tensor async [%dep] %mem, %dims : index, index into memref<?xf64>
```
"""
function create_dn_tensor(asyncDependencies, memref, dims; dnTensor::IR.Type, asyncToken=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[dnTensor, ]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(memref), get_value.(dims)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([length(asyncDependencies), 1, length(dims), ]))
    !isnothing(asyncToken) && push!(results, asyncToken)
    
    create_operation(
        "gpu.create_dn_tensor", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`dealloc`

The `gpu.dealloc` operation frees the region of memory referenced by a
memref which was originally created by the `gpu.alloc` operation. It is
similar to the `memref.dealloc` op, but supports asynchronous GPU execution.

The op does not execute before all async dependencies have finished
executing.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token.

# Example

```mlir
%token = gpu.dealloc async [%dep] %memref : memref<8x64xf32, 1>
```
"""
function dealloc(asyncDependencies, memref; asyncToken=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(memref), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncToken) && push!(results, asyncToken)
    
    create_operation(
        "gpu.dealloc", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`destroy_dn_tensor`

The `gpu.destroy_dn_tensor` operation releases all resources of a dense
tensor represented by a handle that was previously created by a
`gpu.create_dn_tensor` operation.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token in addition to the environment.

# Example

```mlir
%token = gpu.destroy_dn_tensor async [%dep] %dnTensor
```
"""
function destroy_dn_tensor(asyncDependencies, dnTensor; asyncToken=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(dnTensor), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncToken) && push!(results, asyncToken)
    
    create_operation(
        "gpu.destroy_dn_tensor", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`destroy_sp_mat`

The `gpu.destroy_sp_mat` operation releases all resources of a sparse
matrix represented by a handle that was previously created by a
one of the sparse matrix creation operations.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token in addition to the environment.

# Example

```mlir
%token = gpu.destroy_sp_mat async [%dep] %spmat
```
"""
function destroy_sp_mat(asyncDependencies, spmat; asyncToken=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(spmat), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncToken) && push!(results, asyncToken)
    
    create_operation(
        "gpu.destroy_sp_mat", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`dynamic_shared_memory`

This operation provides a memref pointer to the start of dynamic shared
memory, often referred to as workgroup memory. It\'s important to note that
this dynamic shared memory needs to be allocated at kernel launch. One can
conveniently utilize `the dynamic_shared_memory_size` parameter of
`gpu.launch` for this purpose.

Examples:
```mlir
%0 = gpu.dynamic.shared.memory : memref<?xi8, #gpu.address_space<workgroup>>
%1 = memref.view %0[%c8192][] : memref<?xi8, #gpu.address_space<workgroup>>
                        to memref<32x64xf32, #gpu.address_space<workgroup>>
%2 = memref.view %0[%c16384][] : memref<?xi8, #gpu.address_space<workgroup>>
                        to memref<32x64xf32, #gpu.address_space<workgroup>>
```
"""
function dynamic_shared_memory(; resultMemref::IR.Type, location=Location())
    results = IR.Type[resultMemref, ]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "gpu.dynamic_shared_memory", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`func`

Defines a function that can be executed on a GPU. This supports memory
attribution and its body has a particular execution model.

GPU functions are either kernels (as indicated by the `kernel` attribute) or
regular functions. The former can be launched from the host side, while the
latter are device side only.

The memory attribution defines SSA values that correspond to memory buffers
allocated in the memory hierarchy of the GPU (see below).

The operation has one attached region that corresponds to the body of the
function. The region arguments consist of the function arguments without
modification, followed by buffers defined in memory annotations. The body of
a GPU function, when launched, is executed by multiple work items. There are
no guarantees on the order in which work items execute, or on the connection
between them. In particular, work items are not necessarily executed in
lock-step. Synchronization ops such as \"gpu.barrier\" should be used to
coordinate work items. Declarations of GPU functions, i.e. not having the
body region, are not supported.

A function may optionally be annotated with the block and/or grid sizes
that will be used when it is launched using the `gpu.known_block_size` and
`gpu.known_grid_size` attributes, respectively. If set, these attributes must
be arrays of three 32-bit integers giving the x, y, and z launch dimensions.
Launching a kernel that has these annotations, or that calls a function with
these annotations, using a block size or grid size other than what is specified
is undefined behavior.

# Syntax

```
op ::= `gpu.func` symbol-ref-id `(` argument-list `)` (`->`
function-result-list)?
       memory-attribution `kernel`? function-attributes? region

memory-attribution ::= (`workgroup` `(` ssa-id-and-type-list `)`)?
                       (`private` `(` ssa-id-and-type-list `)`)?
```

# Example

```mlir
gpu.func @foo(%arg0: index)
    workgroup(%workgroup: memref<32xf32, 3>)
    private(%private: memref<1xf32, 5>)
    kernel
    attributes {qux: \"quux\"} {
  gpu.return
}
```

The generic form illustrates the concept

```mlir
\"gpu.func\"(%arg: index) {sym_name: \"foo\", kernel, qux: \"quux\"} ({
^bb0(%arg0: index, %workgroup: memref<32xf32, 3>,
     %private: memref<1xf32, 5>):
  \"gpu.return\"() : () -> ()
}) : (index) -> ()
```

Note the non-default memory spaces used in memref types in memory
attribution.
"""
function func(; function_type, arg_attrs=nothing, res_attrs=nothing, workgroup_attrib_attrs=nothing, private_attrib_attrs=nothing, body::Region, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("function_type", function_type), ]
    !isnothing(arg_attrs) && push!(attributes, namedattribute("arg_attrs", arg_attrs))
    !isnothing(res_attrs) && push!(attributes, namedattribute("res_attrs", res_attrs))
    !isnothing(workgroup_attrib_attrs) && push!(attributes, namedattribute("workgroup_attrib_attrs", workgroup_attrib_attrs))
    !isnothing(private_attrib_attrs) && push!(attributes, namedattribute("private_attrib_attrs", private_attrib_attrs))
    
    create_operation(
        "gpu.func", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`module_`

GPU module contains code that is intended to be run on a GPU. A host device
can launch this code through a gpu.launc_func that creates a fully
qualified symbol through the gpu.module\'s symbol and a gpu.func symbol
contained in the gpu.module.

The module\'s top-level scope is modeled by a single region with a single
block. GPU modules are required to have a name that is used for symbol
resolution by the gpu.launch_func operation.

Using an op with a region to define a GPU module enables \"embedding\" GPU
modules with SIMT execution models in other dialects in a clean manner and
allows filtering of code regions to execute passes on only code intended to
or not intended to be run on the separate device.

Modules can contain zero or more target attributes. These attributes encode
how to transform modules into binary strings and are used by the
`gpu-module-to-binary` pass to transform modules into GPU binaries.

Modules can contain an optional `OffloadingTranslationAttr` attribute. This
attribute will be used during the `gpu-module-to-binary` pass to specify the
`OffloadingTranslationAttr` used when creating the `gpu.binary` operation.

```
gpu.module @symbol_name {
  gpu.func {}
    ...
  gpu.module_end
}
// Module with offloading handler and target attributes.
gpu.module @symbol_name2 <#gpu.select_object<1>> [
    #nvvm.target,
    #rocdl.target<chip = \"gfx90a\">] {
  gpu.func {}
    ...
  gpu.module_end
}
```
"""
function module_(; targets=nothing, offloadingHandler=nothing, bodyRegion::Region, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[]
    owned_regions = Region[bodyRegion, ]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(targets) && push!(attributes, namedattribute("targets", targets))
    !isnothing(offloadingHandler) && push!(attributes, namedattribute("offloadingHandler", offloadingHandler))
    
    create_operation(
        "gpu.module", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`global_id`

Returns the unique global workitem/thread id, i.e., the unique index of the
current workitem/thread within all workgroups / grid along the x, y, or z
`dimension`.

# Example

```mlir
%gidX = gpu.global_id x
```
"""
function global_id(; result_0=nothing::Union{Nothing, IR.Type}, dimension, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension), ]
    !isnothing(result_0) && push!(results, result_0)
    
    create_operation(
        "gpu.global_id", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`grid_dim`

Returns the number of thread blocks in the grid along the x, y, or z
`dimension`.

# Example

```mlir
%gDimZ = gpu.grid_dim z
```
"""
function grid_dim(; result_0=nothing::Union{Nothing, IR.Type}, dimension, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension), ]
    !isnothing(result_0) && push!(results, result_0)
    
    create_operation(
        "gpu.grid_dim", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`host_register`

This op maps the provided host buffer into the device address space.

This operation may not be supported in every environment, there is not yet a
way to check at runtime whether this feature is supported.

Writes from the host are guaranteed to be visible to device kernels that are
launched afterwards. Writes from the device are guaranteed to be visible on
the host after synchronizing with the device kernel completion.
"""
function host_register(value; location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value(value), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "gpu.host_register", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`host_unregister`

This op unmaps the provided host buffer from the device address space.

This operation may not be supported in every environment, there is not yet a
    way to check at runtime whether this feature is supported.
"""
function host_unregister(value; location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value(value), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "gpu.host_unregister", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`lane_id`

Returns the lane id within the subgroup (warp/wave).

# Example
```mlir
%laneId = gpu.lane_id
```
"""
function lane_id(; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    create_operation(
        "gpu.lane_id", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`launch_func`

Launch a kernel function on the specified grid of thread blocks.
`gpu.launch` operations are lowered to `gpu.launch_func` operations by
outlining the kernel body into a function in a dedicated module, which
reflects the separate compilation process. The kernel function is required
to have the `gpu.kernel` attribute. The module containing the kernel
function is required to be a gpu.module. And finally, the module containing
the kernel module (which thus cannot be the top-level module) is required
to have the `gpu.container_module` attribute. The `gpu.launch_func`
operation has a symbol attribute named `kernel` to identify the fully
specified kernel function to launch (both the gpu.module and func).

The `gpu.launch_func` supports async dependencies: the kernel does not start
executing until the ops producing those async dependencies have completed.

By the default, the host implicitly blocks until kernel execution has
completed. If the `async` keyword is present, the host does not block but
instead a `!gpu.async.token` is returned. Other async GPU ops can take this
token as dependency.

The operation requires at least the grid and block sizes along the x,y,z
dimensions as arguments. When a lower-dimensional kernel is required,
unused sizes must be explicitly set to `1`.

The remaining operands are optional. The first optional operand corresponds
to the amount of dynamic shared memory a kernel\'s workgroup should be
allocated; when this operand is not present, a zero size is assumed.

The remaining operands if present are passed as arguments to the kernel
function.

The `gpu.launch_func` also supports kernel launching with clusters if
supported by the target architecture. The cluster size can be set by
`clusterSizeX`, `clusterSizeY`, and `clusterSizeZ` arguments. When these
arguments are present, the Op launches a kernel that clusters the given
thread blocks. This feature is exclusive to certain architectures.

# Example

```mlir
module attributes {gpu.container_module} {

  // This module creates a separate compilation unit for the GPU compiler.
  gpu.module @kernels {
    func.func @kernel_1(%arg0 : f32, %arg1 : memref<?xf32, 1>)
        attributes { nvvm.kernel = true } {

      // Operations that produce block/thread IDs and dimensions are
      // injected when outlining the `gpu.launch` body to a function called
      // by `gpu.launch_func`.
      %tIdX = gpu.thread_id x
      %tIdY = gpu.thread_id y
      %tIdZ = gpu.thread_id z

      %bDimX = gpu.block_dim x
      %bDimY = gpu.block_dim y
      %bDimZ = gpu.block_dim z

      %bIdX = gpu.block_id x
      %bIdY = gpu.block_id y
      %bIdZ = gpu.block_id z

      %gDimX = gpu.grid_dim x
      %gDimY = gpu.grid_dim y
      %gDimZ = gpu.grid_dim z

      // (Optional)  Cluster size only for support architectures
      %cIdX = gpu.cluster_id x
      %cIdY = gpu.cluster_id y
      %cIdZ = gpu.cluster_id z

      %cDimX = gpu.cluster_dim x
      %cDimY = gpu.cluster_dim y
      %cDimZ = gpu.cluster_dim z

      \"some_op\"(%bx, %tx) : (index, index) -> ()
      %42 = load %arg1[%bx] : memref<?xf32, 1>
    }
  }

  %t0 = gpu.wait async
  gpu.launch_func
      async                           // (Optional) Don\'t block host, return token.
      [%t0]                           // (Optional) Execute only after %t0 has completed.
      @kernels::@kernel_1             // Kernel function.
      clusters in (%cst, %cst, %cst)  // (Optional) Cluster size only for support architectures.
      blocks in (%cst, %cst, %cst)    // Grid size.
      threads in (%cst, %cst, %cst)   // Block size.
      dynamic_shared_memory_size %s   // (Optional) Amount of dynamic shared
                                      // memory to allocate for a workgroup.
      args(%arg0 : f32,               // (Optional) Kernel arguments.
           %arg1 : memref<?xf32, 1>)
}
```
"""
function launch_func(asyncDependencies, gridSizeX, gridSizeY, gridSizeZ, blockSizeX, blockSizeY, blockSizeZ, clusterSizeX=nothing; clusterSizeY=nothing, clusterSizeZ=nothing, dynamicSharedMemorySize=nothing, kernelOperands, asyncObject=nothing, asyncToken=nothing::Union{Nothing, IR.Type}, kernel, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(gridSizeX), get_value(gridSizeY), get_value(gridSizeZ), get_value(blockSizeX), get_value(blockSizeY), get_value(blockSizeZ), get_value.(kernelOperands)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("kernel", kernel), ]
    (clusterSizeX != nothing) && push!(operands, get_value(clusterSizeX))
    (clusterSizeY != nothing) && push!(operands, get_value(clusterSizeY))
    (clusterSizeZ != nothing) && push!(operands, get_value(clusterSizeZ))
    (dynamicSharedMemorySize != nothing) && push!(operands, get_value(dynamicSharedMemorySize))
    (asyncObject != nothing) && push!(operands, get_value(asyncObject))
    push!(attributes, operandsegmentsizes([length(asyncDependencies), 1, 1, 1, 1, 1, 1, (clusterSizeX==nothing) ? 0 : 1(clusterSizeY==nothing) ? 0 : 1(clusterSizeZ==nothing) ? 0 : 1(dynamicSharedMemorySize==nothing) ? 0 : 1length(kernelOperands), (asyncObject==nothing) ? 0 : 1]))
    !isnothing(asyncToken) && push!(results, asyncToken)
    
    create_operation(
        "gpu.launch_func", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`launch`

Launch a kernel on the specified grid of thread blocks. The body of the
kernel is defined by the single region that this operation contains. The
operation takes an optional list of async dependencies followed by six
operands and an optional operand.

The `async` keyword indicates the kernel should be launched asynchronously;
the operation returns a new !gpu.async.token when the keyword is specified.
The kernel launched does not start executing until the ops producing its
async dependencies (optional operands) have completed.

The first three operands (following any async dependencies) are grid sizes
along the x,y,z dimensions and the following three are block sizes along the
x,y,z dimensions. When a lower-dimensional kernel is required, unused sizes
must be explicitly set to `1`.  The last operand is optional and corresponds
to the amount of dynamic shared memory a kernel\'s workgroup should be
allocated; when this operand is not present, a zero size is assumed.

The body region has at least _twelve_ arguments, or _eighteen_ if cluster 
dimensions are present, grouped as follows:

-   three optional arguments that contain cluster identifiers along x,y,z
    dimensions;
-   three arguments that contain block identifiers along x,y,z dimensions;
-   three arguments that contain thread identifiers along x,y,z dimensions;
-   operands of the `gpu.launch` operation as is (i.e. the operands for
    grid and block sizes).
-   a variadic number of Workgroup memory attributions.
-   a variadic number of Private memory attributions.

# Syntax

```
operation ::= `gpu.launch` (`async` (`[` ssa-id-list `]`)? )?
                         ( `clusters` `(` ssa-id-list `)` `in` ssa-reassignment )?
                         `blocks` `(` ssa-id-list `)` `in` ssa-reassignment
                         `threads` `(` ssa-id-list `)` `in` ssa-reassignment
                         (dynamic_shared_memory_size ssa-use)?
                         memory-attribution
                         region attr-dict?
ssa-reassignment ::= `(` ssa-id `=` ssa-use (`,` ssa-id `=` ssa-use)* `)`
memory-attribution ::= (`workgroup` `(` ssa-id-and-type-list `)`)?
                       (`private` `(` ssa-id-and-type-list `)`)?
```

# Example

```mlir
gpu.launch blocks(%bx, %by, %bz) in (%sz_bx = %0, %sz_by = %1, %sz_bz = %2)
           threads(%tx, %ty, %tz) in (%sz_tx = %3, %sz_ty = %4, %sz_tz = %5) {
  // Block and thread identifiers, as well as block/grid sizes are
  // immediately usable inside body region.
  \"some_op\"(%bx, %tx) : (index, index) -> ()
  // Assuming %val1 is defined outside the gpu.launch region.
  %42 = load %val1[%bx] : memref<?xf32, 1>
}

// Generic syntax explains how the pretty syntax maps to the IR structure.
\"gpu.launch\"(%cst, %cst, %c1,  // Grid sizes.
             %cst, %c1, %c1)   // Block sizes.

    {/*attributes*/}
    // All sizes and identifiers have \"index\" size.
    : (index, index, index, index, index, index) -> () {
// The operation passes block and thread identifiers, followed by grid and
// block sizes.
^bb0(%bx : index, %by : index, %bz : index,
     %tx : index, %ty : index, %tz : index,
     %num_bx : index, %num_by : index, %num_bz : index,
     %num_tx : index, %num_ty : index, %num_tz : index)
  \"some_op\"(%bx, %tx) : (index, index) -> ()
  %3 = \"memref.load\"(%val1, %bx) : (memref<?xf32, 1>, index) -> f32
}

// Launch with memory attributions.
gpu.launch blocks(%bx, %by, %bz) in (%sz_bx = %0, %sz_by = %1, %sz_bz = %2)
           threads(%tx, %ty, %tz) in (%sz_tx = %3, %sz_ty = %4, %sz_tz = %5)
           workgroup(%workgroup: memref<32xf32, 3>)
           private(%private: memref<1xf32, 5>) {
  // Block and thread identifiers, as well as block/grid sizes are
  // immediately usable inside body region.
  \"some_op\"(%bx, %tx) : (index, index) -> ()
  // Assuming %val1 is defined outside the gpu.launch region.
  %42 = load %workgroup[%bx] : memref<32xf32, 3>
}

// Launch with clusters.
gpu.launch clusters(%cx, %cy, %cz) in (%sz_cx = %0, %sz_cy = %1, %sz_cz = %2)
           blocks(%bx, %by, %bz) in (%sz_bx = %3, %sz_by = %4, %sz_bz = %5)
           threads(%tx, %ty, %tz) in (%sz_tx = %6, %sz_ty = %7, %sz_tz = %8)
{
  // Cluster, block and thread identifiers, as well as cluster/block/grid 
  // sizes are immediately usable inside body region.
  \"some_op\"(%cx, %bx, %tx) : (index, index, index) -> ()
}
```

Rationale: using operation/block arguments gives analyses a clear way of
understanding that a value has additional semantics (e.g., we will need to
know what value corresponds to threadIdx.x for coalescing). We can recover
these properties by analyzing the operations producing values, but it is
easier just to have that information by construction.
"""
function launch(asyncDependencies, gridSizeX, gridSizeY, gridSizeZ, blockSizeX, blockSizeY, blockSizeZ, clusterSizeX=nothing; clusterSizeY=nothing, clusterSizeZ=nothing, dynamicSharedMemorySize=nothing, asyncToken=nothing::Union{Nothing, IR.Type}, body::Region, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(gridSizeX), get_value(gridSizeY), get_value(gridSizeZ), get_value(blockSizeX), get_value(blockSizeY), get_value(blockSizeZ), ]
    owned_regions = Region[body, ]
    successors = Block[]
    attributes = NamedAttribute[]
    (clusterSizeX != nothing) && push!(operands, get_value(clusterSizeX))
    (clusterSizeY != nothing) && push!(operands, get_value(clusterSizeY))
    (clusterSizeZ != nothing) && push!(operands, get_value(clusterSizeZ))
    (dynamicSharedMemorySize != nothing) && push!(operands, get_value(dynamicSharedMemorySize))
    push!(attributes, operandsegmentsizes([length(asyncDependencies), 1, 1, 1, 1, 1, 1, (clusterSizeX==nothing) ? 0 : 1(clusterSizeY==nothing) ? 0 : 1(clusterSizeZ==nothing) ? 0 : 1(dynamicSharedMemorySize==nothing) ? 0 : 1]))
    !isnothing(asyncToken) && push!(results, asyncToken)
    
    create_operation(
        "gpu.launch", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`memcpy`

The `gpu.memcpy` operation copies the content of one memref to another.

The op does not execute before all async dependencies have finished
executing.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token.

# Example

```mlir
%token = gpu.memcpy async [%dep] %dst, %src : memref<?xf32, 1>, memref<?xf32>
```
"""
function memcpy(asyncDependencies, dst, src; asyncToken=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(dst), get_value(src), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncToken) && push!(results, asyncToken)
    
    create_operation(
        "gpu.memcpy", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`memset`

The `gpu.memset` operation sets the content of memref to a scalar value.

The op does not execute before all async dependencies have finished
executing.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token.

# Example

```mlir
%token = gpu.memset async [%dep] %dst, %value : memref<?xf32, 1>, f32
```
"""
function memset(asyncDependencies, dst, value; asyncToken=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(dst), get_value(value), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncToken) && push!(results, asyncToken)
    
    create_operation(
        "gpu.memset", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`module_end`

This op terminates the only block inside the only region of a `gpu.module`.
"""
function module_end(; location=Location())
    results = IR.Type[]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "gpu.module_end", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`num_subgroups`

Returns the number of subgroups within a workgroup.

# Example

```mlir
%numSg = gpu.num_subgroups : index
```
"""
function num_subgroups(; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    create_operation(
        "gpu.num_subgroups", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`printf`

`gpu.printf` takes a literal format string `format` and an arbitrary number of
scalar arguments that should be printed.

The format string is a C-style printf string, subject to any restrictions
imposed by one\'s target platform.
"""
function printf(args; format, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value.(args)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("format", format), ]
    
    create_operation(
        "gpu.printf", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`return_`

A terminator operation for regions that appear in the body of  `gpu.func`
functions. The operands to the `gpu.return` are the result values returned
by an invocation of the `gpu.func`.
"""
function return_(operands; location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value.(operands)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "gpu.return", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`sddmm_buffer_size`

The `gpu.sddmm_buffer_size` operation returns the buffer size required
to perform the SDDMM operation on the given sparse and dense matrices.
The operation expects handles returned by previous sparse operations
to construct an environment and the operands for SDDMM.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token in addition to the environment.

# Example

```mlir
%buffersz, %token = gpu.sddmm_buffer_size async [%dep] %dnmatA{TRANSPOSE}, %dnmatB{TRANSPOSE}, %spmatC into f32
```

The matrix arguments can also be associated with one of the following
operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value
is NON_TRANSPOSE.
"""
function sddmm_buffer_size(asyncDependencies, dnmatA, dnmatB, spmatC; bufferSz::IR.Type, asyncToken=nothing::Union{Nothing, IR.Type}, modeA=nothing, modeB=nothing, computeType, location=Location())
    results = IR.Type[bufferSz, ]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(dnmatA), get_value(dnmatB), get_value(spmatC), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("computeType", computeType), ]
    !isnothing(asyncToken) && push!(results, asyncToken)
    !isnothing(modeA) && push!(attributes, namedattribute("modeA", modeA))
    !isnothing(modeB) && push!(attributes, namedattribute("modeB", modeB))
    
    create_operation(
        "gpu.sddmm_buffer_size", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`sddmm`

The `gpu.sddmm` operation performs the SDDMM operation on the given sparse and
dense matrices, and buffer.  The operation expects handles returned by previous
sparse operations to construct an environment and the operands for SDDMM. The
buffer must have been allocated on the device.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token in addition to the environment.

# Example

```mlir
%token = gpu.sddmm async [%dep] %dnmatA{TRANSPOSE}, %dnmatB{TRANSPOSE}, %spmatC, %buffer into f32
```

The matrix arguments can also be associated with one of the following
operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value
is NON_TRANSPOSE.
"""
function sddmm(asyncDependencies, dnmatA, dnmatB, spmatC, buffer; asyncToken=nothing::Union{Nothing, IR.Type}, modeA=nothing, modeB=nothing, computeType, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(dnmatA), get_value(dnmatB), get_value(spmatC), get_value(buffer), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("computeType", computeType), ]
    !isnothing(asyncToken) && push!(results, asyncToken)
    !isnothing(modeA) && push!(attributes, namedattribute("modeA", modeA))
    !isnothing(modeB) && push!(attributes, namedattribute("modeB", modeB))
    
    create_operation(
        "gpu.sddmm", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`set_csr_pointers`

The `gpu.set_csr_pointers` assigns the given positions, coordinates,
and values buffer that reside on the device directly to the given sparse
matrix descriptor in csr format.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a `!gpu.async.token` in addition to the environment.

# Example

```mlir
%token = gpu.set_csr_pointers async [%dep] %positions, %coordinates, %values
      : memref<?xf32>, memref<?xindex>, memref<?xindex>
```
"""
function set_csr_pointers(asyncDependencies, spmat, positions, coordinates, values; asyncToken=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(spmat), get_value(positions), get_value(coordinates), get_value(values), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncToken) && push!(results, asyncToken)
    
    create_operation(
        "gpu.set_csr_pointers", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`set_default_device`

Operation that sets the current default GPU, using a zero-based index
into the set of GPUs on the system. The default GPU setting may be
thread-local.
"""
function set_default_device(devIndex; location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value(devIndex), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "gpu.set_default_device", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`shuffle`

The \"shuffle\" op moves values to a different invocation within the same
subgroup.

# Example

```mlir
%1, %2 = gpu.shuffle %0, %offset, %width xor : f32
```

For lane k returns the value from lane `k ^ offset` and `true` if that lane
is smaller than %width. Otherwise it returns an unspecified value and
`false`. A lane is the index of an invocation relative to its subgroup.

The width specifies the number of invocations that participate in the
shuffle. The width needs to be the same for all invocations that participate
in the shuffle. Exactly the first `width` invocations of a subgroup need to
execute this op in convergence.
"""
function shuffle(value, offset, width; shuffleResult=nothing::Union{Nothing, IR.Type}, valid=nothing::Union{Nothing, IR.Type}, mode, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value(value), get_value(offset), get_value(width), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("mode", mode), ]
    !isnothing(shuffleResult) && push!(results, shuffleResult)
    !isnothing(valid) && push!(results, valid)
    
    create_operation(
        "gpu.shuffle", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`spgemm_copy`

The `gpu.spgemm_copy` operation copies the sparse matrix result of
a SpGEMM computation.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a `!gpu.async.token` in addition to the environment.

# Example

```mlir
gpu.spgemm_copy %spmatA, %spmatB, %spmatC, %spgemmDesc: f32
```

The matrix arguments can also be associated with one of the following
operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value
is NON_TRANSPOSE.
"""
function spgemm_copy(asyncDependencies, desc, spmatA, spmatB, spmatC; asyncToken=nothing::Union{Nothing, IR.Type}, modeA=nothing, modeB=nothing, computeType, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(desc), get_value(spmatA), get_value(spmatB), get_value(spmatC), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("computeType", computeType), ]
    !isnothing(asyncToken) && push!(results, asyncToken)
    !isnothing(modeA) && push!(attributes, namedattribute("modeA", modeA))
    !isnothing(modeB) && push!(attributes, namedattribute("modeB", modeB))
    
    create_operation(
        "gpu.spgemm_copy", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`spgemm_create_descr`

The `gpu.spgemm_create_descr` creates a descriptor for the SpGEMM operation.
The descriptor describes the SpGEMM operation and stores the internal data
throughout the computation. It needs to be passed as an argument to
spgemm_* operations.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a `!gpu.async.token` in addition to the environment.

# Example

```mlir
%desc, %token = gpu.spgemm_create_descr async [%dep]
```
"""
function spgemm_create_descr(asyncDependencies; desc::IR.Type, asyncToken=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[desc, ]
    operands = API.MlirValue[get_value.(asyncDependencies)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncToken) && push!(results, asyncToken)
    
    create_operation(
        "gpu.spgemm_create_descr", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`spgemm_destroy_descr`

The `gpu.spgemm_destroy_descr` destroys the SpGEMM operation descriptor.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a `!gpu.async.token` in addition to the environment.

# Example

```mlir
%token = gpu.spgemm_destroy_descr async [%dep] %desc
```
"""
function spgemm_destroy_descr(asyncDependencies, desc; asyncToken=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(desc), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncToken) && push!(results, asyncToken)
    
    create_operation(
        "gpu.spgemm_destroy_descr", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`spgemm_work_estimation_or_compute`

The `gpu.spgemm_work_estimation_or_compute` is used to call
cusparseSpGEMM_workEstimation or cusparseSpGEMM_compute. Both of them are
for both determining the buffer size and performing the actual computation.
The operation expects handles returned by previous sparse operations to
construct an environment and the operands for SpGEMM.
The buffer must have been allocated on the device.

C\' = alpha * op(A) * op(B) + beta * C

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a `!gpu.async.token` in addition to the environment.

# Example

```mlir
%bufferSz, %token = gpu.spgemm_work_estimation_or_compute async [%dep] {COMPUTE}
                      %desc, %spmatA{NON_TRANSPOSE}, %spmatB{NON_TRANSPOSE},
                      %spmatC, %spgemmDesc, %c0, %alloc: f32 into
                      memref<0xi8>
```

The matrix arguments can also be associated with one of the following
operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value
is NON_TRANSPOSE.
"""
function spgemm_work_estimation_or_compute(asyncDependencies, desc, spmatA, spmatB, spmatC, bufferSz, buffer; bufferSzNew::IR.Type, asyncToken=nothing::Union{Nothing, IR.Type}, modeA=nothing, modeB=nothing, computeType, kind, location=Location())
    results = IR.Type[bufferSzNew, ]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(desc), get_value(spmatA), get_value(spmatB), get_value(spmatC), get_value(bufferSz), get_value(buffer), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("computeType", computeType), namedattribute("kind", kind), ]
    !isnothing(asyncToken) && push!(results, asyncToken)
    !isnothing(modeA) && push!(attributes, namedattribute("modeA", modeA))
    !isnothing(modeB) && push!(attributes, namedattribute("modeB", modeB))
    
    create_operation(
        "gpu.spgemm_work_estimation_or_compute", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`spmm_buffer_size`

The `gpu.spmm_buffer_size` operation returns the buffer size required
to perform the SpMM operation on the given sparse and dense matrix.
The operation expects handles returned by previous sparse operations
to construct an environment and the operands for SpMM.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token in addition to the environment.

The matrix arguments can also be associated with one of the following
operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value
is NON_TRANSPOSE.

# Example

```mlir
%bufferszs, %token = gpu.spmm_buffer_size async [%dep] %spmatA{TRANSPOSE}, %dnmatB{TRANSPOSE}, %dnmatC : i64 into f32
```
"""
function spmm_buffer_size(asyncDependencies, spmatA, dnmatB, dnmatC; bufferSzs::Vector{IR.Type}, asyncToken=nothing::Union{Nothing, IR.Type}, modeA=nothing, modeB=nothing, computeType, location=Location())
    results = IR.Type[bufferSzs..., ]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(spmatA), get_value(dnmatB), get_value(dnmatC), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("computeType", computeType), ]
    !isnothing(asyncToken) && push!(results, asyncToken)
    !isnothing(modeA) && push!(attributes, namedattribute("modeA", modeA))
    !isnothing(modeB) && push!(attributes, namedattribute("modeB", modeB))
    
    create_operation(
        "gpu.spmm_buffer_size", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`spmm`

The `gpu.spmm` operation performs the SpMM operation on the given sparse and
dense matrix, and buffer.  The operation expects handles returned by previous
sparse operations to construct an environment and the operands for SpMM. The
buffer must have been allocated on the device.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token in addition to the environment.

The matrix arguments can also be associated with one of the following
operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value
is NON_TRANSPOSE.

# Example

```mlir
%token = gpu.spmm async [%dep] %spmatA{TRANSPOSE}, %dnmatB{TRANSPOSE}, %dnmatC, %buffers : type(\$buffers) into f32
```
"""
function spmm(asyncDependencies, spmatA, dnmatB, dnmatC, buffers; asyncToken=nothing::Union{Nothing, IR.Type}, modeA=nothing, modeB=nothing, computeType, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(spmatA), get_value(dnmatB), get_value(dnmatC), get_value.(buffers)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("computeType", computeType), ]
    push!(attributes, operandsegmentsizes([length(asyncDependencies), 1, 1, 1, length(buffers), ]))
    !isnothing(asyncToken) && push!(results, asyncToken)
    !isnothing(modeA) && push!(attributes, namedattribute("modeA", modeA))
    !isnothing(modeB) && push!(attributes, namedattribute("modeB", modeB))
    
    create_operation(
        "gpu.spmm", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`spmv_buffer_size`

The `gpu.spmv_buffer_size` operation returns the buffer size required
to perform the SpMV operation on the given sparse matrix and dense vectors.
The operation expects handles returned by previous sparse operations
to construct an environment and the operands for SpMV.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token in addition to the environment.

The matrix arguments can also be associated with one of the following
operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value
is NON_TRANSPOSE.

# Example

```mlir
%buffersz, %token = gpu.spmv_buffer_size async [%dep] %spmatA{TRANSPOSE}, %dnX, %dnY into f32
```
"""
function spmv_buffer_size(asyncDependencies, spmatA, dnX, dnY; bufferSz::IR.Type, asyncToken=nothing::Union{Nothing, IR.Type}, modeA=nothing, computeType, location=Location())
    results = IR.Type[bufferSz, ]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(spmatA), get_value(dnX), get_value(dnY), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("computeType", computeType), ]
    !isnothing(asyncToken) && push!(results, asyncToken)
    !isnothing(modeA) && push!(attributes, namedattribute("modeA", modeA))
    
    create_operation(
        "gpu.spmv_buffer_size", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`spmv`

The `gpu.spmv` operation performs the SpMV operation on the given sparse matrix,
dense vectors, and buffer.  The operation expects handles returned by previous
sparse operations to construct an environment and the operands for SpMV. The
buffer must have been allocated on the device.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token in addition to the environment.

The matrix arguments can also be associated with one of the following
operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value
is NON_TRANSPOSE.

# Example

```mlir
%token = gpu.spmv async [%dep] %spmatA{TRANSPOSE}, %dnX, %dnY : memref<?xf64> into bf16
```
"""
function spmv(asyncDependencies, spmatA, dnX, dnY, buffer; asyncToken=nothing::Union{Nothing, IR.Type}, modeA=nothing, computeType, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(spmatA), get_value(dnX), get_value(dnY), get_value(buffer), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("computeType", computeType), ]
    !isnothing(asyncToken) && push!(results, asyncToken)
    !isnothing(modeA) && push!(attributes, namedattribute("modeA", modeA))
    
    create_operation(
        "gpu.spmv", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`spmat_get_size`

The `gpu.spmat_get_size` operation retrieves the number of rows, number of
columns, and number of non-zero elements of a sparse matrix.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a `!gpu.async.token` in addition to the environment.

# Example

```mlir
%rows, %cols, %nnz, %token = gpu.spmat_get_size async [%dep] %spmatC
```
"""
function spmat_get_size(asyncDependencies, spmat; rows::IR.Type, cols::IR.Type, nnz::IR.Type, asyncToken=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[rows, cols, nnz, ]
    operands = API.MlirValue[get_value.(asyncDependencies)..., get_value(spmat), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncToken) && push!(results, asyncToken)
    
    create_operation(
        "gpu.spmat_get_size", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`subgroup_id`

Returns the subgroup id, i.e. the index of the current subgroup within the
workgroup.

# Example

```mlir
%sgId = gpu.subgroup_id : index
```
"""
function subgroup_id(; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    create_operation(
        "gpu.subgroup_id", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`subgroup_mma_compute`

The `gpu.subgroup_mma_compute` operation performs a matrix-multiply accumulate (mma)
operation using all the threads in a subgroup.

This operation takes three `!gpu.mma_matrix`s as arguments: these hold `A`,
`B` and `C`operands for the mma operation. The operation performed is represented
as `C += A * B`. The op returns a `!gpu.mma_matrix` which contains the result of
the operation held by all threads in a subgroup. `a_transpose` or
`b_transpose` if present, signify that the respective operand was loaded in a
transposed manner. The transpose operands are required to map to correct
underlying intrisics but they currently do not seem to affect correctness
even if they are absent given that the operands were loaded correctly using
the `transpose` attribute in `gpu.subgroup_mma_load_matrix` op.

For integer types, the `A` and `B` matrices carry their signedness with their
types. The accumulator type is expected to be signless and imply a signed integer
with a greater width than the other two operands.

This op is meant to be used along with `gpu.subgroup_mma_store_matrix` and
`gpu.subgroup_mma_load_matrix` ops.

# Example

```mlir
%D = gpu.subgroup_mma_compute_matrix %A, %B, %C :
  !gpu.mma_matrix<16x16xf16, \"AOp\">, !gpu.mma_matrix<16x16xf16, \"BOp\">>
  -> !gpu.mma_matrix<16x16xf16, \"COp\">
```
"""
function subgroup_mma_compute(opA, opB, opC; res=nothing::Union{Nothing, IR.Type}, a_transpose=nothing, b_transpose=nothing, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value(opA), get_value(opB), get_value(opC), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(res) && push!(results, res)
    !isnothing(a_transpose) && push!(attributes, namedattribute("a_transpose", a_transpose))
    !isnothing(b_transpose) && push!(attributes, namedattribute("b_transpose", b_transpose))
    
    create_operation(
        "gpu.subgroup_mma_compute", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`subgroup_mma_constant_matrix`

The `gpu.subgroup_mma_constant_matrix` creates a `!gpu.mma_matrix` with
constant elements.

The operation takes a scalar input and return a `!gpu.mma_matrix` where
each element of is equal to the operand constant. The destination
mma_matrix type must have elememt type equal to the constant type. Since
the layout of `!gpu.mma_matrix` is opaque this only support setting all the
elements to the same value.

This op is meant to be used along with `gpu.subgroup_mma_compute`.

# Example

```mlir
 %0 = gpu.subgroup_mma_constant_matrix %a :
   !gpu.mma_matrix<16x16xf16, \"AOp\">
 %1 = gpu.subgroup_mma_constant_matrix %b :
   !gpu.mma_matrix<16x16xf32, \"COp\">
```
"""
function subgroup_mma_constant_matrix(value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = API.MlirValue[get_value(value), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "gpu.subgroup_mma_constant_matrix", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`subgroup_mma_elementwise`

The `gpu.subgroup_mma_elementwise` takes `!gpu.mma_matrix` inputs and
compute a new `!gpu.mma_matrix` by applying an elementwise operation to each
element.

Since the operation is elementwise and the matrix type must match, the
matrix elements are processed independently of the matrix layout.

This op is meant to be used along with `gpu.subgroup_mma_compute`.

# Example

```mlir
 %0 =  %A, %B { opType = \"ADD\" } :
  (!gpu.mma_matrix<16x16xf16, \"COp\">, !gpu.mma_matrix<16x16xf16, \"COp\">)
  -> !gpu.mma_matrix<16x16xf16, \"COp\">
```
"""
function subgroup_mma_elementwise(args; res::IR.Type, opType, location=Location())
    results = IR.Type[res, ]
    operands = API.MlirValue[get_value.(args)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("opType", opType), ]
    
    create_operation(
        "gpu.subgroup_mma_elementwise", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`subgroup_mma_load_matrix`

The `gpu.subgroup_mma_load_matrix` operation loads a matrix collectively
using all the threads in a subgroup.

This operation takes a memref as its first operand: it is the source matrix
from which data is to be loaded. The op returns a `!gpu.mma_matrix`. The
source memref can be in global memory or shared memory. The load address is
determined using `indices`. The matrix being loaded into is the result.  The
`leadDimension` attribute specifies the leading dimension size of the source
matrix which eventually allows the lowering to determine the size of each
row.  If the `transpose` attribute is present then the op does a transposed load.

For integer types, the resulting `!gpu.mma_matrix` type needs to specify the
signedness of the data if the matrix type is an `A` or `B` operand for
`gpu.subgroup_mma_compute`.

This op is often meant to be used along with `gpu.subgroup_mma_store_matrix` and
`gpu.subgroup_mma_compute`.

# Example

```mlir
 %0 = gpu.subgroup_mma_load_matrix src[%i,%j] : {leadDimension = 32 : i32}
      : memref<32x32xf16, 3>, !gpu.mma_matrix<16x16xf16, \"AOp\">
```
"""
function subgroup_mma_load_matrix(srcMemref, indices; res::IR.Type, leadDimension, transpose=nothing, location=Location())
    results = IR.Type[res, ]
    operands = API.MlirValue[get_value(srcMemref), get_value.(indices)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("leadDimension", leadDimension), ]
    !isnothing(transpose) && push!(attributes, namedattribute("transpose", transpose))
    
    create_operation(
        "gpu.subgroup_mma_load_matrix", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`subgroup_mma_store_matrix`

The `gpu.subgroup_mma_store_matrix` operation stores a matrix collectively
using all the threads in a subgroup.

This operation takes a `!gpu.mma_matrix` and a memref as operands.
`!gpu.mma_matrix` is the source value containing the data to be stored into the
destination memref which can be in global or shared memory.  The store address
is determined using the indices provided. The `leadDimension` attribute
specifies the leading dimension of the destination matrix. If the
`transpose` attribute is present then the op does a transposed store.

This op is often meant to be used along with `gpu.subgroup_mma_load_matrix` and
`gpu.subgroup_mma_compute`.

# Example

```mlir
gpu.subgroup_mma_store_matrix %D, %sg[%i,%j] : { leadDimension = 32 : i32}
                : !gpu.mma_matrix<16x16xf16, \"COp\">, memref<32x32xf16, 3>
```
"""
function subgroup_mma_store_matrix(src, dstMemref, indices; leadDimension, transpose=nothing, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value(src), get_value(dstMemref), get_value.(indices)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("leadDimension", leadDimension), ]
    !isnothing(transpose) && push!(attributes, namedattribute("transpose", transpose))
    
    create_operation(
        "gpu.subgroup_mma_store_matrix", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`subgroup_reduce`

The `subgroup_reduce` op reduces the value of every work item across a
subgroup. The result is equal for all work items of a subgroup.

When the reduced value is of a vector type, each vector element is reduced
independently. Only 1-d vector types are allowed.

# Example

```mlir
%1 = gpu.subgroup_reduce add %a : (f32) -> (f32)
%2 = gpu.subgroup_reduce add %b : (vector<4xf16>) -> (vector<4xf16>)
```

If `uniform` flag is set either none or all work items of a subgroup
need to execute this op in convergence. The reduction operation must be one
of:
*  Integer types: `add`, `mul`, `minui`, `minsi`, `maxui`, `maxsi`, `and`,
   `or`, `xor`
*  Floating point types: `add`, `mul`, `minnumf`, `maxnumf`, `minimumf`,
   `maximumf`
"""
function subgroup_reduce(value; result=nothing::Union{Nothing, IR.Type}, op, uniform=nothing, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value(value), ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("op", op), ]
    !isnothing(result) && push!(results, result)
    !isnothing(uniform) && push!(attributes, namedattribute("uniform", uniform))
    
    create_operation(
        "gpu.subgroup_reduce", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`subgroup_size`

Returns the number of threads within a subgroup.

# Example

```mlir
%sgSz = gpu.subgroup_size : index
```
"""
function subgroup_size(; result=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(result) && push!(results, result)
    
    create_operation(
        "gpu.subgroup_size", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`terminator`

A terminator operation for regions that appear in the body of `gpu.launch`
operation.  These regions are not expected to return any value so the
terminator takes no operands.
"""
function terminator(; location=Location())
    results = IR.Type[]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "gpu.terminator", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`thread_id`

Returns the thread id, i.e. the index of the current thread within the block
along the x, y, or z `dimension`.

# Example

```mlir
%tIdX = gpu.thread_id x
```
"""
function thread_id(; result_0=nothing::Union{Nothing, IR.Type}, dimension, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("dimension", dimension), ]
    !isnothing(result_0) && push!(results, result_0)
    
    create_operation(
        "gpu.thread_id", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`wait`

This op synchronizes the host or the device with a list of dependent ops.

If the op contains the `async` keyword, it returns a new async token which
is synchronized with the op arguments. This new token is merely a shortcut
to the argument list, and one could replace the uses of the result with the
arguments for the same effect. The async version of this op is primarily
used to make each async token have a single use during lowering and
thereby make forks in async execution explicit. Example usage:

```mlir
%t0 = gpu.foo async : !gpu.async.token
%t1 = gpu.bar async : !gpu.async.token
%t2 = gpu.wait async [%t0, %t1]
// gpu.baz doesn\'t run until gpu.foo and gpu.bar have both completed, just
// as if the async dependencies were [%t0, %t1].
%t3 = gpu.baz async [%t2]
```

If the op does not contain the `async` keyword, it does not return a new
async token but blocks until all ops producing the async dependency tokens
finished execution. All dependent memory operations are visible to the host
once this op completes. Example usage:

```mlir
%t0 = gpu.foo async : !gpu.async.token
%t1 = gpu.bar async : !gpu.async.token
// The gpu.wait op blocks until gpu.foo and gpu.bar have completed.
gpu.wait [%t0, %t1]
```
"""
function wait(asyncDependencies; asyncToken=nothing::Union{Nothing, IR.Type}, location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value.(asyncDependencies)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    !isnothing(asyncToken) && push!(results, asyncToken)
    
    create_operation(
        "gpu.wait", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`yield`

gpu.yield` is a special terminator operation for blocks inside regions
in gpu ops. It returns values to the immediately enclosing gpu op.

# Example

```mlir
gpu.yield %f0, %f1 : f32, f32
```
"""
function yield(values; location=Location())
    results = IR.Type[]
    operands = API.MlirValue[get_value.(values)..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "gpu.yield", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # gpu
