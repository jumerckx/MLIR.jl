module Gpu

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation, context
import ...API
import ..Dialects: make_named_attribute

"""
all_reduce

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
accumulation as code region. The accumulation operation must be one of:
`add`, `and`, `max`, `min`, `mul`, `or`, `xor`.

If `uniform` flag is set either none or all work items of a workgroup
need to execute this op in convergence.
  
"""
function AllReduce(; location::Location, _unnamed0_::MLIRType, value_::Value, body_::Region, op_=nothing::Union{Nothing,Union{NamedAttribute,Attribute}}, uniform_=nothing::Union{Nothing,Union{NamedAttribute,Attribute}})
    results = [_unnamed0_]
    operands = [value_]
    regions = [body_]
    successors = []
    attributes = []

    (op_ != nothing) && push!(attributes, make_named_attribute("op", op_))
    (uniform_ != nothing) && push!(attributes, make_named_attribute("uniform", uniform_))

    create_operation(
        "gpu.all_reduce", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
alloc

The `gpu.alloc` operation allocates a region of memory on the GPU. It is
similar to the `memref.alloc` op, but supports asynchronous GPU execution.

The op does not execute before all async dependencies have finished
executing.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it also returns a !gpu.async.token.

If the `host_shared` keyword is present, the memory will be allocated in a
memory accessible both on host and on device.

Example:

```mlir
%memref, %token = gpu.alloc async [%dep] host_shared (%width) : memref<64x?xf32, 1>
```
  
"""
function Alloc(; location::Location, memref_::MLIRType, asyncToken_=nothing::Union{Nothing,MLIRType}, asyncDependencies_::Vector{Value}, dynamicSizes_::Vector{Value}, symbolOperands_::Vector{Value}, hostShared_=nothing::Union{Nothing,Union{NamedAttribute,Attribute}})
    results = [memref_]
    (asyncToken_ != nothing) && push!(results, asyncToken_)
    operands = [asyncDependencies_..., dynamicSizes_..., symbolOperands_...]
    regions = []
    successors = []
    attributes = []

    (hostShared_ != nothing) && push!(attributes, make_named_attribute("hostShared", hostShared_))

    push!(attributes, make_named_attribute("operand_segment_sizes", Int32[]))

    create_operation(
        "gpu.alloc", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
barrier

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
function Barrier(; location::Location)
    results = []
    operands = []
    regions = []
    successors = []
    attributes = []

    create_operation(
        "gpu.barrier", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
block_dim

Returns the number of threads in the thread block (aka the block size) along
the x, y, or z `dimension`.

Example:

```mlir
%bDimX = gpu.block_dim x
```
  
"""
function BlockDim(; location::Location, _unnamed0_::MLIRType, dimension_::Union{NamedAttribute,Attribute})
    results = [_unnamed0_]
    operands = []
    regions = []
    successors = []
    attributes = [make_named_attribute("dimension", dimension_)]

    create_operation(
        "gpu.block_dim", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
block_id

Returns the block id, i.e. the index of the current block within the grid
along the x, y, or z `dimension`.

Example:

```mlir
%bIdY = gpu.block_id y
```
  
"""
function BlockId(; location::Location, _unnamed0_::MLIRType, dimension_::Union{NamedAttribute,Attribute})
    results = [_unnamed0_]
    operands = []
    regions = []
    successors = []
    attributes = [make_named_attribute("dimension", dimension_)]

    create_operation(
        "gpu.block_id", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
create_2to4_spmat

The `gpu.create_2to4_spmat` operation initializes a sparse matrix in dense
format with 2:4 sparsity.
The buffers must already be copied from the host to the device prior to
using this operation. The operation returns a handle to the sparse
matrix descriptor.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token in addition to the environment.

Example:

```mlir
%spmat, %token = gpu.create_2to4_spmat async [%dep] {PRUNE_AND_CHECK} %rows, %cols, %mem: memref<?xf64>
```
  
"""
function Create2To4SpMat(; location::Location, spMat_::MLIRType, asyncToken_=nothing::Union{Nothing,MLIRType}, asyncDependencies_::Vector{Value}, rows_::Value, cols_::Value, memref_::Value, pruneFlag_=nothing::Union{Nothing,Union{NamedAttribute,Attribute}})
    results = [spMat_]
    (asyncToken_ != nothing) && push!(results, asyncToken_)
    operands = [asyncDependencies_..., rows_, cols_, memref_]
    regions = []
    successors = []
    attributes = []

    (pruneFlag_ != nothing) && push!(attributes, make_named_attribute("pruneFlag", pruneFlag_))

    create_operation(
        "gpu.create_2to4_spmat", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
create_coo_aos

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

Example:

```mlir
%spmat, %token = gpu.create_coo_aos async [%dep] %rows, %cols, %nnz, %idxs,
    %values : memref<?xindex>, memref<?xf64>
```
  
"""
function CreateCooAoS(; location::Location, spmat_::MLIRType, asyncToken_=nothing::Union{Nothing,MLIRType}, asyncDependencies_::Vector{Value}, rows_::Value, cols_::Value, nnz_::Value, idxs_::Value, values_::Value)
    results = [spmat_]
    (asyncToken_ != nothing) && push!(results, asyncToken_)
    operands = [asyncDependencies_..., rows_, cols_, nnz_, idxs_, values_]
    regions = []
    successors = []
    attributes = []

    create_operation(
        "gpu.create_coo_aos", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
create_coo

The `gpu.create_coo` operation initializes a sparse matrix in COO format
with the given sizes from the given index and values buffers. The buffers
must already be copied from the host to the device prior to using this
operation. The operation returns a handle to the sparse matrix descriptor.
Note that this operation builds the COO in SoA format.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token in addition to the environment.

Example:

```mlir
%spmat, %token = gpu.create_coo async [%dep] %rows, %cols, %nnz, %rowIdx,
    %colIdx, %values : memref<?xindex>, memref<?xindex>, memref<?xf64>
```
  
"""
function CreateCoo(; location::Location, spmat_::MLIRType, asyncToken_=nothing::Union{Nothing,MLIRType}, asyncDependencies_::Vector{Value}, rows_::Value, cols_::Value, nnz_::Value, rowIdxs_::Value, colIdxs_::Value, values_::Value)
    results = [spmat_]
    (asyncToken_ != nothing) && push!(results, asyncToken_)
    operands = [asyncDependencies_..., rows_, cols_, nnz_, rowIdxs_, colIdxs_, values_]
    regions = []
    successors = []
    attributes = []

    create_operation(
        "gpu.create_coo", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
create_csr

The `gpu.create_csr` operation initializes a sparse matrix in CSR format
with the given sizes from the given position, index, and values buffers.
The buffers must already be copied from the host to the device prior to
using this operation. The operation returns a handle to the sparse
matrix descriptor.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token in addition to the environment.

Example:

```mlir
%spmat, %token = gpu.create_csr async [%dep] %rows, %cols, %nnz, %rowPos,
    %colIdx, %values : memref<?xindex>, memref<?xindex>, memref<?xf64>
```
  
"""
function CreateCsr(; location::Location, spmat_::MLIRType, asyncToken_=nothing::Union{Nothing,MLIRType}, asyncDependencies_::Vector{Value}, rows_::Value, cols_::Value, nnz_::Value, rowPos_::Value, colIdxs_::Value, values_::Value)
    results = [spmat_]
    (asyncToken_ != nothing) && push!(results, asyncToken_)
    operands = [asyncDependencies_..., rows_, cols_, nnz_, rowPos_, colIdxs_, values_]
    regions = []
    successors = []
    attributes = []

    create_operation(
        "gpu.create_csr", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
create_dn_tensor

The `gpu.create_dn_tensor` operation initializes a dense tensor from
the given values buffer and sizes. The buffer must already be copied
from the host to the device prior to using this operation. The
operation returns a handle to the dense tensor descriptor.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token in addition to the environment.

Example:

```mlir
%dmat, %token = gpu.create_dn_tensor async [%dep] %mem, %dims : index, index into memref<?xf64>
```
  
"""
function CreateDnTensor(; location::Location, dnTensor_::MLIRType, asyncToken_=nothing::Union{Nothing,MLIRType}, asyncDependencies_::Vector{Value}, memref_::Value, dims_::Vector{Value})
    results = [dnTensor_]
    (asyncToken_ != nothing) && push!(results, asyncToken_)
    operands = [asyncDependencies_..., memref_, dims_...]
    regions = []
    successors = []
    attributes = []

    push!(attributes, make_named_attribute("operand_segment_sizes", Int32[]))

    create_operation(
        "gpu.create_dn_tensor", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
dealloc

The `gpu.dealloc` operation frees the region of memory referenced by a
memref which was originally created by the `gpu.alloc` operation. It is
similar to the `memref.dealloc` op, but supports asynchronous GPU execution.

The op does not execute before all async dependencies have finished
executing.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token.

Example:

```mlir
%token = gpu.dealloc async [%dep] %memref : memref<8x64xf32, 1>
```
  
"""
function Dealloc(; location::Location, asyncToken_=nothing::Union{Nothing,MLIRType}, asyncDependencies_::Vector{Value}, memref_::Value)
    results = []
    (asyncToken_ != nothing) && push!(results, asyncToken_)
    operands = [asyncDependencies_..., memref_]
    regions = []
    successors = []
    attributes = []

    create_operation(
        "gpu.dealloc", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
destroy_dn_tensor

The `gpu.destroy_dn_tensor` operation releases all resources of a dense
tensor represented by a handle that was previously created by a
`gpu.create_dn_tensor` operation.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token in addition to the environment.

Example:

```mlir
%token = gpu.destroy_dn_tensor async [%dep] %dnTensor
```
  
"""
function DestroyDnTensor(; location::Location, asyncToken_=nothing::Union{Nothing,MLIRType}, asyncDependencies_::Vector{Value}, dnTensor_::Value)
    results = []
    (asyncToken_ != nothing) && push!(results, asyncToken_)
    operands = [asyncDependencies_..., dnTensor_]
    regions = []
    successors = []
    attributes = []

    create_operation(
        "gpu.destroy_dn_tensor", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
destroy_sp_mat

The `gpu.destroy_sp_mat` operation releases all resources of a sparse
matrix represented by a handle that was previously created by a
one of the sparse matrix creation operations.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token in addition to the environment.

Example:

```mlir
%token = gpu.destroy_sp_mat async [%dep] %spmat
```
  
"""
function DestroySpMat(; location::Location, asyncToken_=nothing::Union{Nothing,MLIRType}, asyncDependencies_::Vector{Value}, spmat_::Value)
    results = []
    (asyncToken_ != nothing) && push!(results, asyncToken_)
    operands = [asyncDependencies_..., spmat_]
    regions = []
    successors = []
    attributes = []

    create_operation(
        "gpu.destroy_sp_mat", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
func

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

Syntax:

```
op ::= `gpu.func` symbol-ref-id `(` argument-list `)` (`->`
function-result-list)?
       memory-attribution `kernel`? function-attributes? region

memory-attribution ::= (`workgroup` `(` ssa-id-and-type-list `)`)?
                       (`private` `(` ssa-id-and-type-list `)`)?
```

Example:

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
function GPUFunc(; location::Location, body_::Region, function_type_::Union{NamedAttribute,Attribute}, arg_attrs_=nothing::Union{Nothing,Union{NamedAttribute,Attribute}}, res_attrs_=nothing::Union{Nothing,Union{NamedAttribute,Attribute}}, workgroup_attrib_attrs_=nothing::Union{Nothing,Union{NamedAttribute,Attribute}}, private_attrib_attrs_=nothing::Union{Nothing,Union{NamedAttribute,Attribute}})
    results = []
    operands = []
    regions = [body_]
    successors = []
    attributes = [make_named_attribute("function_type", function_type_)]

    (arg_attrs_ != nothing) && push!(attributes, make_named_attribute("arg_attrs", arg_attrs_))
    (res_attrs_ != nothing) && push!(attributes, make_named_attribute("res_attrs", res_attrs_))
    (workgroup_attrib_attrs_ != nothing) && push!(attributes, make_named_attribute("workgroup_attrib_attrs", workgroup_attrib_attrs_))
    (private_attrib_attrs_ != nothing) && push!(attributes, make_named_attribute("private_attrib_attrs", private_attrib_attrs_))

    create_operation(
        "gpu.func", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
module

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

```
  gpu.module @symbol_name {
  gpu.func {}
    ...
  gpu.module_end
}

```
  
"""
function GPUModule(; location::Location, bodyRegion_::Region, sym_name_::NamedAttribute)
    results = []
    operands = []
    regions = [bodyRegion_]
    successors = []
    attributes = [sym_name_]

    create_operation(
        "gpu.module", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
global_id

Returns the unique global workitem/thread id, i.e., the unique index of the
current workitem/thread within all workgroups / grid along the x, y, or z
`dimension`.

Example:

```mlir
%gidX = gpu.global_id x
```
  
"""
function GlobalId(; location::Location, _unnamed0_::MLIRType, dimension_::Union{NamedAttribute,Attribute})
    results = [_unnamed0_]
    operands = []
    regions = []
    successors = []
    attributes = [make_named_attribute("dimension", dimension_)]

    create_operation(
        "gpu.global_id", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
grid_dim

Returns the number of thread blocks in the grid along the x, y, or z
`dimension`.

Example:

```mlir
%gDimZ = gpu.grid_dim z
```
  
"""
function GridDim(; location::Location, _unnamed0_::MLIRType, dimension_::Union{NamedAttribute,Attribute})
    results = [_unnamed0_]
    operands = []
    regions = []
    successors = []
    attributes = [make_named_attribute("dimension", dimension_)]

    create_operation(
        "gpu.grid_dim", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
host_register

This op maps the provided host buffer into the device address space.

This operation may not be supported in every environment, there is not yet a
way to check at runtime whether this feature is supported.

Writes from the host are guaranteed to be visible to device kernels that are
launched afterwards. Writes from the device are guaranteed to be visible on
the host after synchronizing with the device kernel completion.
  
"""
function HostRegister(; location::Location, value_::Value)
    results = []
    operands = [value_]
    regions = []
    successors = []
    attributes = []

    create_operation(
        "gpu.host_register", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
host_unregister

This op unmaps the provided host buffer from the device address space.

This operation may not be supported in every environment, there is not yet a
    way to check at runtime whether this feature is supported.
  
"""
function HostUnregister(; location::Location, value_::Value)
    results = []
    operands = [value_]
    regions = []
    successors = []
    attributes = []

    create_operation(
        "gpu.host_unregister", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
lane_id

Returns the lane id within the subgroup (warp/wave).

Example:
```mlir
%laneId = gpu.lane_id
```
  
"""
function LaneId(; location::Location, result_::MLIRType)
    results = [result_]
    operands = []
    regions = []
    successors = []
    attributes = []

    create_operation(
        "gpu.lane_id", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
launch_func

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

Example:

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

      \"some_op\"(%bx, %tx) : (index, index) -> ()
      %42 = load %arg1[%bx] : memref<?xf32, 1>
    }
  }

  %t0 = gpu.wait async
  gpu.launch_func
      async                           // (Optional) Don\'t block host, return token.
      [%t0]                           // (Optional) Execute only after %t0 has completed.
      @kernels::@kernel_1             // Kernel function.
      blocks in (%cst, %cst, %cst)    // Grid size.
      threads in (%cst, %cst, %cst)   // Block size.
      dynamic_shared_memory_size %s   // (Optional) Amount of dynamic shared
                                      // memory to allocate for a workgroup.
      args(%arg0 : f32,               // (Optional) Kernel arguments.
           %arg1 : memref<?xf32, 1>)
}
```
  
"""
function LaunchFunc(; location::Location, asyncToken_=nothing::Union{Nothing,MLIRType}, asyncDependencies_::Vector{Value}, gridSizeX_::Value, gridSizeY_::Value, gridSizeZ_::Value, blockSizeX_::Value, blockSizeY_::Value, blockSizeZ_::Value, dynamicSharedMemorySize_=nothing::Union{Nothing,Value}, kernelOperands_::Vector{Value}, kernel_::Union{NamedAttribute,Attribute})
    results = []
    (asyncToken_ != nothing) && push!(results, asyncToken_)
    operands = [asyncDependencies_..., gridSizeX_, gridSizeY_, gridSizeZ_, blockSizeX_, blockSizeY_, blockSizeZ_, kernelOperands_...]

    (dynamicSharedMemorySize_ != nothing) && push!(operands, dynamicSharedMemorySize_)
    regions = []
    successors = []
    attributes = [make_named_attribute("kernel", kernel_)]

    push!(attributes, make_named_attribute("operand_segment_sizes", Int32[]))

    create_operation(
        "gpu.launch_func", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
launch

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

The body region has at least _twelve_ arguments, grouped as follows:

-   three arguments that contain block identifiers along x,y,z dimensions;
-   three arguments that contain thread identifiers along x,y,z dimensions;
-   operands of the `gpu.launch` operation as is (i.e. the operands for
    grid and block sizes).
-   a variadic number of Workgroup memory attributions.
-   a variadic number of Private memory attributions.

Syntax:

```
operation ::= `gpu.launch` (`async` (`[` ssa-id-list `]`)? )?
                         `block` `(` ssa-id-list `)` `in` ssa-reassignment
                         `threads` `(` ssa-id-list `)` `in` ssa-reassignment
                         (dynamic_shared_memory_size ssa-use)?
                         memory-attribution
                         region attr-dict?
ssa-reassignment ::= `(` ssa-id `=` ssa-use (`,` ssa-id `=` ssa-use)* `)`
memory-attribution ::= (`workgroup` `(` ssa-id-and-type-list `)`)?
                       (`private` `(` ssa-id-and-type-list `)`)?
```

Example:

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
```

Rationale: using operation/block arguments gives analyses a clear way of
understanding that a value has additional semantics (e.g., we will need to
know what value corresponds to threadIdx.x for coalescing). We can recover
these properties by analyzing the operations producing values, but it is
easier just to have that information by construction.
  
"""
function Launch(; location::Location, asyncToken_=nothing::Union{Nothing,MLIRType}, asyncDependencies_::Vector{Value}, gridSizeX_::Value, gridSizeY_::Value, gridSizeZ_::Value, blockSizeX_::Value, blockSizeY_::Value, blockSizeZ_::Value, dynamicSharedMemorySize_=nothing::Union{Nothing,Value}, body_::Region)
    results = []
    (asyncToken_ != nothing) && push!(results, asyncToken_)
    operands = Value[asyncDependencies_..., gridSizeX_, gridSizeY_, gridSizeZ_, blockSizeX_, blockSizeY_, blockSizeZ_]

    (dynamicSharedMemorySize_ != nothing) && push!(operands, dynamicSharedMemorySize_)
    regions = Region[body_]
    successors = Block[]
    attributes = NamedAttribute[]

    push!(attributes, NamedAttribute("operand_segment_sizes",
        Attribute(API.mlirDenseI32ArrayGet(
            context().context,
            8,
            Int32[0, 1, 1, 1, 1, 1, 1, 0])))
    )

    # push!(attributes, make_named_attribute("operand_segment_sizes", Int32[]))

    create_operation(
        "gpu.launch", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
memcpy

The `gpu.memcpy` operation copies the content of one memref to another.

The op does not execute before all async dependencies have finished
executing.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token.

Example:

```mlir
%token = gpu.memcpy async [%dep] %dst, %src : memref<?xf32, 1>, memref<?xf32>
```
  
"""
function Memcpy(; location::Location, asyncToken_=nothing::Union{Nothing,MLIRType}, asyncDependencies_::Vector{Value}, dst_::Value, src_::Value)
    results = []
    (asyncToken_ != nothing) && push!(results, asyncToken_)
    operands = [asyncDependencies_..., dst_, src_]
    regions = []
    successors = []
    attributes = []

    create_operation(
        "gpu.memcpy", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
memset

The `gpu.memset` operation sets the content of memref to a scalar value.

The op does not execute before all async dependencies have finished
executing.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token.

Example:

```mlir
%token = gpu.memset async [%dep] %dst, %value : memref<?xf32, 1>, f32
```
  
"""
function Memset(; location::Location, asyncToken_=nothing::Union{Nothing,MLIRType}, asyncDependencies_::Vector{Value}, dst_::Value, value_::Value)
    results = []
    (asyncToken_ != nothing) && push!(results, asyncToken_)
    operands = [asyncDependencies_..., dst_, value_]
    regions = []
    successors = []
    attributes = []

    create_operation(
        "gpu.memset", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
module_end

This op terminates the only block inside the only region of a `gpu.module`.
  
"""
function ModuleEnd(; location::Location)
    results = []
    operands = []
    regions = []
    successors = []
    attributes = []

    create_operation(
        "gpu.module_end", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
num_subgroups

Returns the number of subgroups within a workgroup.

Example:

```mlir
%numSg = gpu.num_subgroups : index
```
  
"""
function NumSubgroups(; location::Location, result_::MLIRType)
    results = [result_]
    operands = []
    regions = []
    successors = []
    attributes = []

    create_operation(
        "gpu.num_subgroups", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
printf

`gpu.printf` takes a literal format string `format` and an arbitrary number of
scalar arguments that should be printed.

The format string is a C-style printf string, subject to any restrictions
imposed by one\'s target platform.
  
"""
function Printf(; location::Location, args_::Vector{Value}, format_::Union{NamedAttribute,String})
    results = []
    operands = [args_...]
    regions = []
    successors = []
    attributes = [make_named_attribute("format", format_)]

    create_operation(
        "gpu.printf", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
return

A terminator operation for regions that appear in the body of  `gpu.func`
functions. The operands to the `gpu.return` are the result values returned
by an invocation of the `gpu.func`.
  
"""
function Return(; location::Location, operands_::Vector{Value})
    results = []
    operands = [operands_...]
    regions = []
    successors = []
    attributes = []

    create_operation(
        "gpu.return", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
sddmm_buffer_size

The `gpu.sddmm_buffer_size` operation returns the buffer size required
to perform the SDDMM operation on the given sparse and dense matrices.
The operation expects handles returned by previous sparse operations
to construct an environment and the operands for SDDMM.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token in addition to the environment.

Example:

```mlir
%buffersz, %token = gpu.sddmm_buffer_size async [%dep] %dnmatA{TRANSPOSE}, %dnmatB{TRANSPOSE}, %spmatC into f32
```

The matrix arguments can also be associated with one of the following
operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value
is NON_TRANSPOSE.
  
"""
function SDDMMBufferSize(; location::Location, bufferSz_::MLIRType, asyncToken_=nothing::Union{Nothing,MLIRType}, asyncDependencies_::Vector{Value}, dnmatA_::Value, dnmatB_::Value, spmatC_::Value, modeA_=nothing::Union{Nothing,Union{NamedAttribute,Attribute}}, modeB_=nothing::Union{Nothing,Union{NamedAttribute,Attribute}}, computeType_::Union{NamedAttribute,Attribute})
    results = [bufferSz_]
    (asyncToken_ != nothing) && push!(results, asyncToken_)
    operands = [asyncDependencies_..., dnmatA_, dnmatB_, spmatC_]
    regions = []
    successors = []
    attributes = [make_named_attribute("computeType", computeType_)]

    (modeA_ != nothing) && push!(attributes, make_named_attribute("modeA", modeA_))
    (modeB_ != nothing) && push!(attributes, make_named_attribute("modeB", modeB_))

    create_operation(
        "gpu.sddmm_buffer_size", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
sddmm

The `gpu.sddmm` operation performs the SDDMM operation on the given sparse and
dense matrices, and buffer.  The operation expects handles returned by previous
sparse operations to construct an environment and the operands for SDDMM. The
buffer must have been allocated on the device.

If the `async` keyword is present, the op is executed asynchronously (i.e.
it does not block until the execution has finished on the device). In
that case, it returns a !gpu.async.token in addition to the environment.

Example:

```mlir
%token = gpu.sddmm async [%dep] %dnmatA{TRANSPOSE}, %dnmatB{TRANSPOSE}, %spmatC, %buffer into f32
```

The matrix arguments can also be associated with one of the following
operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value
is NON_TRANSPOSE.
  
"""
function SDDMM(; location::Location, asyncToken_=nothing::Union{Nothing,MLIRType}, asyncDependencies_::Vector{Value}, dnmatA_::Value, dnmatB_::Value, spmatC_::Value, buffer_::Value, modeA_=nothing::Union{Nothing,Union{NamedAttribute,Attribute}}, modeB_=nothing::Union{Nothing,Union{NamedAttribute,Attribute}}, computeType_::Union{NamedAttribute,Attribute})
    results = []
    (asyncToken_ != nothing) && push!(results, asyncToken_)
    operands = [asyncDependencies_..., dnmatA_, dnmatB_, spmatC_, buffer_]
    regions = []
    successors = []
    attributes = [make_named_attribute("computeType", computeType_)]

    (modeA_ != nothing) && push!(attributes, make_named_attribute("modeA", modeA_))
    (modeB_ != nothing) && push!(attributes, make_named_attribute("modeB", modeB_))

    create_operation(
        "gpu.sddmm", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
set_default_device

Operation that sets the current default GPU, using a zero-based index
into the set of GPUs on the system. The default GPU setting may be
thread-local.
  
"""
function SetDefaultDevice(; location::Location, devIndex_::Value)
    results = []
    operands = [devIndex_]
    regions = []
    successors = []
    attributes = []

    create_operation(
        "gpu.set_default_device", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
shuffle

The \"shuffle\" op moves values to a different invocation within the same
subgroup.

Example:

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
function Shuffle(; location::Location, shuffleResult_::MLIRType, valid_::MLIRType, value_::Value, offset_::Value, width_::Value, mode_::Union{NamedAttribute,Attribute})
    results = [shuffleResult_, valid_]
    operands = [value_, offset_, width_]
    regions = []
    successors = []
    attributes = [make_named_attribute("mode", mode_)]

    create_operation(
        "gpu.shuffle", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
spmm_buffer_size

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

Example:

```mlir
%bufferszs, %token = gpu.spmm_buffer_size async [%dep] %spmatA{TRANSPOSE}, %dnmatB{TRANSPOSE}, %dnmatC : i64 into f32
```
  
"""
function SpMMBufferSize(; location::Location, bufferSzs_::Vector{MLIRType}, asyncToken_=nothing::Union{Nothing,MLIRType}, asyncDependencies_::Vector{Value}, spmatA_::Value, dnmatB_::Value, dnmatC_::Value, modeA_=nothing::Union{Nothing,Union{NamedAttribute,Attribute}}, modeB_=nothing::Union{Nothing,Union{NamedAttribute,Attribute}}, computeType_::Union{NamedAttribute,Attribute})
    results = [bufferSzs_...]
    (asyncToken_ != nothing) && push!(results, asyncToken_)
    operands = [asyncDependencies_..., spmatA_, dnmatB_, dnmatC_]
    regions = []
    successors = []
    attributes = [make_named_attribute("computeType", computeType_)]

    (modeA_ != nothing) && push!(attributes, make_named_attribute("modeA", modeA_))
    (modeB_ != nothing) && push!(attributes, make_named_attribute("modeB", modeB_))

    create_operation(
        "gpu.spmm_buffer_size", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
spmm

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

Example:

```mlir
%token = gpu.spmm async [%dep] %spmatA{TRANSPOSE}, %dnmatB{TRANSPOSE}, %dnmatC, %buffers : type(\$buffers) into f32
```
  
"""
function SpMM(; location::Location, asyncToken_=nothing::Union{Nothing,MLIRType}, asyncDependencies_::Vector{Value}, spmatA_::Value, dnmatB_::Value, dnmatC_::Value, buffers_::Vector{Value}, modeA_=nothing::Union{Nothing,Union{NamedAttribute,Attribute}}, modeB_=nothing::Union{Nothing,Union{NamedAttribute,Attribute}}, computeType_::Union{NamedAttribute,Attribute})
    results = []
    (asyncToken_ != nothing) && push!(results, asyncToken_)
    operands = [asyncDependencies_..., spmatA_, dnmatB_, dnmatC_, buffers_...]
    regions = []
    successors = []
    attributes = [make_named_attribute("computeType", computeType_)]

    (modeA_ != nothing) && push!(attributes, make_named_attribute("modeA", modeA_))
    (modeB_ != nothing) && push!(attributes, make_named_attribute("modeB", modeB_))

    push!(attributes, make_named_attribute("operand_segment_sizes", Int32[]))

    create_operation(
        "gpu.spmm", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
spmv_buffer_size

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

Example:

```mlir
%buffersz, %token = gpu.spmv_buffer_size async [%dep] %spmatA{TRANSPOSE}, %dnX, %dnY into f32
```
  
"""
function SpMVBufferSize(; location::Location, bufferSz_::MLIRType, asyncToken_=nothing::Union{Nothing,MLIRType}, asyncDependencies_::Vector{Value}, spmatA_::Value, dnX_::Value, dnY_::Value, modeA_=nothing::Union{Nothing,Union{NamedAttribute,Attribute}}, computeType_::Union{NamedAttribute,Attribute})
    results = [bufferSz_]
    (asyncToken_ != nothing) && push!(results, asyncToken_)
    operands = [asyncDependencies_..., spmatA_, dnX_, dnY_]
    regions = []
    successors = []
    attributes = [make_named_attribute("computeType", computeType_)]

    (modeA_ != nothing) && push!(attributes, make_named_attribute("modeA", modeA_))

    create_operation(
        "gpu.spmv_buffer_size", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
spmv

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

Example:

```mlir
%token = gpu.spmv async [%dep] %spmatA{TRANSPOSE}, %dnX, %dnY : memref<?xf64> into bf16
```
  
"""
function SpMV(; location::Location, asyncToken_=nothing::Union{Nothing,MLIRType}, asyncDependencies_::Vector{Value}, spmatA_::Value, dnX_::Value, dnY_::Value, buffer_::Value, modeA_=nothing::Union{Nothing,Union{NamedAttribute,Attribute}}, computeType_::Union{NamedAttribute,Attribute})
    results = []
    (asyncToken_ != nothing) && push!(results, asyncToken_)
    operands = [asyncDependencies_..., spmatA_, dnX_, dnY_, buffer_]
    regions = []
    successors = []
    attributes = [make_named_attribute("computeType", computeType_)]

    (modeA_ != nothing) && push!(attributes, make_named_attribute("modeA", modeA_))

    create_operation(
        "gpu.spmv", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
subgroup_id

Returns the subgroup id, i.e. the index of the current subgroup within the
workgroup.

Example:

```mlir
%sgId = gpu.subgroup_id : index
```
  
"""
function SubgroupId(; location::Location, result_::MLIRType)
    results = [result_]
    operands = []
    regions = []
    successors = []
    attributes = []

    create_operation(
        "gpu.subgroup_id", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
subgroup_mma_compute

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

Example:

```mlir
%D = gpu.subgroup_mma_compute_matrix %A, %B, %C :
  !gpu.mma_matrix<16x16xf16, \"AOp\">, !gpu.mma_matrix<16x16xf16, \"BOp\">>
  -> !gpu.mma_matrix<16x16xf16, \"COp\">
```
  
"""
function SubgroupMmaCompute(; location::Location, res_::MLIRType, opA_::Value, opB_::Value, opC_::Value, a_transpose_=nothing::Union{Nothing,Union{NamedAttribute,Attribute}}, b_transpose_=nothing::Union{Nothing,Union{NamedAttribute,Attribute}})
    results = [res_]
    operands = [opA_, opB_, opC_]
    regions = []
    successors = []
    attributes = []

    (a_transpose_ != nothing) && push!(attributes, make_named_attribute("a_transpose", a_transpose_))
    (b_transpose_ != nothing) && push!(attributes, make_named_attribute("b_transpose", b_transpose_))

    create_operation(
        "gpu.subgroup_mma_compute", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
subgroup_mma_constant_matrix

The `gpu.subgroup_mma_constant_matrix` creates a `!gpu.mma_matrix` with
constant elements.

The operation takes a scalar input and return a `!gpu.mma_matrix` where
each element of is equal to the operand constant. The destination
mma_matrix type must have elememt type equal to the constant type. Since
the layout of `!gpu.mma_matrix` is opaque this only support setting all the
elements to the same value.

This op is meant to be used along with `gpu.subgroup_mma_compute`.

Example:

```mlir
 %0 = gpu.subgroup_mma_constant_matrix %a :
   !gpu.mma_matrix<16x16xf16, \"AOp\">
 %1 = gpu.subgroup_mma_constant_matrix %b :
   !gpu.mma_matrix<16x16xf32, \"COp\">
```
  
"""
function SubgroupMmaConstantMatrix(; location::Location, res_::MLIRType, value_::Value)
    results = [res_]
    operands = [value_]
    regions = []
    successors = []
    attributes = []

    create_operation(
        "gpu.subgroup_mma_constant_matrix", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
subgroup_mma_elementwise

The `gpu.subgroup_mma_elementwise` takes `!gpu.mma_matrix` inputs and
compute a new `!gpu.mma_matrix` by applying an elementwise operation to each
element.

Since the operation is elementwise and the matrix type must match, the
matrix elements are processed independently of the matrix layout.

This op is meant to be used along with `gpu.subgroup_mma_compute`.

Example:

```mlir
 %0 =  %A, %B { opType = \"ADD\" } :
  (!gpu.mma_matrix<16x16xf16, \"COp\">, !gpu.mma_matrix<16x16xf16, \"COp\">)
  -> !gpu.mma_matrix<16x16xf16, \"COp\">
```
  
"""
function SubgroupMmaElementwise(; location::Location, res_::MLIRType, args_::Vector{Value}, opType_::Union{NamedAttribute,Attribute})
    results = [res_]
    operands = [args_...]
    regions = []
    successors = []
    attributes = [make_named_attribute("opType", opType_)]

    create_operation(
        "gpu.subgroup_mma_elementwise", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
subgroup_mma_load_matrix

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

Example:

```mlir
 %0 = gpu.subgroup_mma_load_matrix src[%i,%j] : {leadDimension = 32 : i32}
      : memref<32x32xf16, 3>, !gpu.mma_matrix<16x16xf16, \"AOp\">
```
  
"""
function SubgroupMmaLoadMatrix(; location::Location, res_::MLIRType, srcMemref_::Value, indices_::Vector{Value}, leadDimension_::Union{NamedAttribute,Attribute}, transpose_=nothing::Union{Nothing,Union{NamedAttribute,Attribute}})
    results = [res_]
    operands = [srcMemref_, indices_...]
    regions = []
    successors = []
    attributes = [make_named_attribute("leadDimension", leadDimension_)]

    (transpose_ != nothing) && push!(attributes, make_named_attribute("transpose", transpose_))

    create_operation(
        "gpu.subgroup_mma_load_matrix", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
subgroup_mma_store_matrix

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

Example:

```mlir
gpu.subgroup_mma_store_matrix %D, %sg[%i,%j] : { leadDimension = 32 : i32}
                : !gpu.mma_matrix<16x16xf16, \"COp\">, memref<32x32xf16, 3>
```
  
"""
function SubgroupMmaStoreMatrix(; location::Location, src_::Value, dstMemref_::Value, indices_::Vector{Value}, leadDimension_::Union{NamedAttribute,Attribute}, transpose_=nothing::Union{Nothing,Union{NamedAttribute,Attribute}})
    results = []
    operands = [src_, dstMemref_, indices_...]
    regions = []
    successors = []
    attributes = [make_named_attribute("leadDimension", leadDimension_)]

    (transpose_ != nothing) && push!(attributes, make_named_attribute("transpose", transpose_))

    create_operation(
        "gpu.subgroup_mma_store_matrix", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
subgroup_reduce

The `subgroup_reduce` op reduces the value of every work item across a
subgroup. The result is equal for all work items of a subgroup.

Example:

```mlir
%1 = gpu.subgroup_reduce add %0 : (f32) -> (f32)
```

If `uniform` flag is set either none or all work items of a subgroup
need to execute this op in convergence.
  
"""
function SubgroupReduce(; location::Location, _unnamed0_::MLIRType, value_::Value, op_::Union{NamedAttribute,Attribute}, uniform_=nothing::Union{Nothing,Union{NamedAttribute,Attribute}})
    results = [_unnamed0_]
    operands = [value_]
    regions = []
    successors = []
    attributes = [make_named_attribute("op", op_)]

    (uniform_ != nothing) && push!(attributes, make_named_attribute("uniform", uniform_))

    create_operation(
        "gpu.subgroup_reduce", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
subgroup_size

Returns the number of threads within a subgroup.

Example:

```mlir
%sgSz = gpu.subgroup_size : index
```
  
"""
function SubgroupSize(; location::Location, result_::MLIRType)
    results = [result_]
    operands = []
    regions = []
    successors = []
    attributes = []

    create_operation(
        "gpu.subgroup_size", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
terminator

A terminator operation for regions that appear in the body of `gpu.launch`
operation.  These regions are not expected to return any value so the
terminator takes no operands.
  
"""
function Terminator(; location::Location)
    results = []
    operands = []
    regions = []
    successors = []
    attributes = []

    create_operation(
        "gpu.terminator", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
thread_id

Returns the thread id, i.e. the index of the current thread within the block
along the x, y, or z `dimension`.

Example:

```mlir
%tIdX = gpu.thread_id x
```
  
"""
function ThreadId(; location::Location, _unnamed0_::MLIRType, dimension_::Union{NamedAttribute,Attribute})
    results = [_unnamed0_]
    operands = []
    regions = []
    successors = []
    attributes = [make_named_attribute("dimension", dimension_)]

    create_operation(
        "gpu.thread_id", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
wait

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
function Wait(; location::Location, asyncToken_=nothing::Union{Nothing,MLIRType}, asyncDependencies_::Vector{Value})
    results = []
    (asyncToken_ != nothing) && push!(results, asyncToken_)
    operands = [asyncDependencies_...]
    regions = []
    successors = []
    attributes = []

    create_operation(
        "gpu.wait", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


"""
yield

gpu.yield` is a special terminator operation for blocks inside regions
in gpu ops. It returns values to the immediately enclosing gpu op.

Example:

```mlir
gpu.yield %f0, %f1 : f32, f32
```
  
"""
function Yield(; location::Location, values_::Vector{Value})
    results = []
    operands = [values_...]
    regions = []
    successors = []
    attributes = []

    create_operation(
        "gpu.yield", location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference=false
    )
end


end #Gpu
