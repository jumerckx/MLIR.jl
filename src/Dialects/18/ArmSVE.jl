module arm_sve

import ...IR: IR, NamedAttribute, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`intr_convert_from_svbool`

"""
function intr_convert_from_svbool(svbool::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[svbool, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.intr.convert.from.svbool", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`convert_from_svbool`

Converts svbool types (`vector<[16]xi1>` or vectors of that type, e.g.
`vector<2x3x[16]xi1>`) to SVE predicate types. Note: Only the trailing
dimension can be scalable.

Example 1: Convert a 1-D svbool mask to a SVE predicate.
```mlir
%source = vector.load %memref[%c0] : memref<?xi1>, vector<[16]xi1>
%result = arm_sve.convert_from_svbool %source : vector<[4]xi1>
```

Example 2: Convert a 2-D svbool mask to a mask of SVE predicates.
```mlir
%source = vector.load %memref[%c0, %c0] : memref<2x?xi1>, vector<2x[16]xi1>
%result = arm_sve.convert_from_svbool %source : vector<2x[8]xi1>
```

---

A `svbool` is the smallest SVE predicate type that has a in-memory
representation (and maps to a full predicate register). In MLIR `svbool` is
represented as `vector<[16]xi1>`. Smaller SVE predicate types
(`vector<[1|2|4|8]xi1>`) must be stored as a `svbool` then converted back to
the original predicate type after loading.
"""
function convert_from_svbool(source::Value; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[source, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.convert_from_svbool", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_convert_to_svbool`

"""
function intr_convert_to_svbool(mask::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[mask, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.intr.convert.to.svbool", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`convert_to_svbool`

Converts SVE predicate types (or vectors of predicate types, e.g.
`vector<4x[4]xi1>`) to svbool types. Note: Only the trailing dimension can
be scalable.

Example 1: Convert a 1-D SVE predicate to a svbool mask.
```mlir
%source = vector.create_mask %dim_size : vector<[4]xi1>
%result = arm_sve.convert_to_svbool %source : vector<[4]xi1>
// => Results in vector<[16]xi1>
```

Example 2: Convert a 2-D mask of SVE predicates to a svbool mask.
```mlir
%source = vector.create_mask %c2, %dim_size : vector<2x[2]xi1>
%result = arm_sve.convert_to_svbool %source : vector<2x[2]xi1>
// => Results in vector<2x[16]xi1>
```

---

A `svbool` is the smallest SVE predicate type that has a in-memory
representation (and maps to a full predicate register). In MLIR `svbool` is
represented as `vector<[16]xi1>`. Smaller SVE predicate types
(`vector<[1|2|4|8]xi1>`) must be converted to a `svbool` before they can be
stored.
"""
function convert_to_svbool(source::Value; result::IR.Type, location=Location())
    results = IR.Type[result, ]
    operands = Value[source, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.convert_to_svbool", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_fadd`

"""
function intr_fadd(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.intr.fadd", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`masked_addf`

The `arm_sve.masked.addf` operation takes one scalable vector mask
and two scalable vector operands, and perform floating point addition on active lanes. Inactive lanes will keep the value of
the first operand.
"""
function masked_addf(mask::Value, src1::Value, src2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[mask, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.masked.addf", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_add`

"""
function intr_add(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.intr.add", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`masked_addi`

The `arm_sve.masked.addi` operation takes one scalable vector mask
and two scalable vector operands, and perform integer addition on active lanes. Inactive lanes will keep the value of
the first operand.
"""
function masked_addi(mask::Value, src1::Value, src2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[mask, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.masked.addi", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_fdiv`

"""
function intr_fdiv(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.intr.fdiv", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`masked_divf`

The `arm_sve.masked.divf` operation takes one scalable vector mask
and two scalable vector operands, and perform floating point division on active lanes. Inactive lanes will keep the value of
the first operand.
"""
function masked_divf(mask::Value, src1::Value, src2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[mask, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.masked.divf", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_fmul`

"""
function intr_fmul(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.intr.fmul", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`masked_mulf`

The `arm_sve.masked.mulf` operation takes one scalable vector mask
and two scalable vector operands, and perform floating point multiplication on active lanes. Inactive lanes will keep the value of
the first operand.
"""
function masked_mulf(mask::Value, src1::Value, src2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[mask, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.masked.mulf", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_mul`

"""
function intr_mul(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.intr.mul", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`masked_muli`

The `arm_sve.masked.muli` operation takes one scalable vector mask
and two scalable vector operands, and perform integer multiplication on active lanes. Inactive lanes will keep the value of
the first operand.
"""
function masked_muli(mask::Value, src1::Value, src2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[mask, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.masked.muli", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_sdiv`

"""
function intr_sdiv(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.intr.sdiv", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`masked_divi_signed`

The `arm_sve.masked.divi_signed` operation takes one scalable vector mask
and two scalable vector operands, and perform integer signed division on active lanes. Inactive lanes will keep the value of
the first operand.
"""
function masked_divi_signed(mask::Value, src1::Value, src2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[mask, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.masked.divi_signed", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_fsub`

"""
function intr_fsub(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.intr.fsub", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`masked_subf`

The `arm_sve.masked.subf` operation takes one scalable vector mask
and two scalable vector operands, and perform floating point subtraction on active lanes. Inactive lanes will keep the value of
the first operand.
"""
function masked_subf(mask::Value, src1::Value, src2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[mask, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.masked.subf", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_sub`

"""
function intr_sub(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.intr.sub", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`masked_subi`

The `arm_sve.masked.subi` operation takes one scalable vector mask
and two scalable vector operands, and perform integer subtraction on active lanes. Inactive lanes will keep the value of
the first operand.
"""
function masked_subi(mask::Value, src1::Value, src2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[mask, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.masked.subi", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_udiv`

"""
function intr_udiv(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.intr.udiv", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`masked_divi_unsigned`

The `arm_sve.masked.divi_unsigned` operation takes one scalable vector mask
and two scalable vector operands, and perform integer unsigned division on active lanes. Inactive lanes will keep the value of
the first operand.
"""
function masked_divi_unsigned(mask::Value, src1::Value, src2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[mask, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.masked.divi_unsigned", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_sdot`

"""
function intr_sdot(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.intr.sdot", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`sdot`

SDOT: Signed integer addition of dot product.

This function maps to the SDOT instruction, and it takes signless integer
operands that the operation interprets as signed. It partitions the second
and third vector inputs into groups of four elements. They calculate the dot
product of each group (without loss of precision) and then add each result
to the overlapping element of the first vector input.

Source:
https://developer.arm.com/documentation/100987/0000
"""
function sdot(acc::Value, src1::Value, src2::Value; dst::IR.Type, location=Location())
    results = IR.Type[dst, ]
    operands = Value[acc, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.sdot", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_smmla`

"""
function intr_smmla(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.intr.smmla", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`smmla`

SMMLA: Signed integer matrix multiply-accumulate.

This function maps to the SMMLA instruction, and it takes signless integer
operands that the operation interprets as signed. It partitions the inputs
into 128-bit quadwords, with the first input containing a row-by-row 2×2
matrix of 32-bit integers, the second input containing a row-by-row 2×8
matrix of 8-bit integers, and the third input containing a column-by-column
8×2 matrix of 8-bit integers. For each quadword, they multiply the second
input matrix by the third input matrix using natural arithmetic and then add
the result to the first input using modular arithmetic.

Source:
https://developer.arm.com/documentation/100987/0000
"""
function smmla(acc::Value, src1::Value, src2::Value; dst::IR.Type, location=Location())
    results = IR.Type[dst, ]
    operands = Value[acc, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.smmla", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_udot`

"""
function intr_udot(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.intr.udot", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`udot`

UDOT: Unsigned integer addition of dot product.

This function maps to the UDOT instruction, and it takes signless integer
operands that the operation interprets as unsigned. It partitions the second
and third vector inputs into groups of four elements. They calculate the dot
product of each group (without loss of precision) and then add each result
to the overlapping element of the first vector input.

Source:
https://developer.arm.com/documentation/100987/0000
"""
function udot(acc::Value, src1::Value, src2::Value; dst::IR.Type, location=Location())
    results = IR.Type[dst, ]
    operands = Value[acc, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.udot", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_ummla`

"""
function intr_ummla(operand_0::Value, operand_1::Value, operand_2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[operand_0, operand_1, operand_2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.intr.ummla", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ummla`

UMMLA: Unsigned integer matrix multiply-accumulate.

This function maps to the UMMLA instruction, and it takes signless integer
operands that the operation interprets as unsigned. It partitions the inputs
into 128-bit quadwords, with the first input containing a row-by-row 2×2
matrix of 32-bit integers, the second input containing a row-by-row 2×8
matrix of 8-bit integers, and the third input containing a column-by-column
8×2 matrix of 8-bit integers. For each quadword, they multiply the second
input matrix by the third input matrix using natural arithmetic and then add
the result to the first input using modular arithmetic.

Source:
https://developer.arm.com/documentation/100987/0000
"""
function ummla(acc::Value, src1::Value, src2::Value; dst::IR.Type, location=Location())
    results = IR.Type[dst, ]
    operands = Value[acc, src1, src2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.ummla", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_zip_x2`

"""
function intr_zip_x2(v1::Value, v2::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[v1, v2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.intr.zip.x2", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`zip_x2`

This operation interleaves elements from two input SVE vectors, returning
two new SVE vectors (`resultV1` and `resultV2`), which contain the low and
high halves of the result respectively.

# Example
```mlir
// sourceV1 = [ A1, A2, A3, ... An ]
// sourceV2 = [ B1, B2, B3, ... Bn ]
// (resultV1, resultV2) = [ A1, B1, A2, B2, A3, B3, ... An, Bn ]
%resultV1, %resultV2 = arm_sve.zip.x2 %sourceV1, %sourceV2 : vector<[16]xi8>
```

Note: This requires SME 2 (`+sme2` in LLVM target features)

[Source](https://developer.arm.com/documentation/ddi0602/2023-12/SME-Instructions/ZIP--two-registers---Interleave-elements-from-two-vectors-?lang=en)
"""
function zip_x2(sourceV1::Value, sourceV2::Value; resultV1::IR.Type, resultV2::IR.Type, location=Location())
    results = IR.Type[resultV1, resultV2, ]
    operands = Value[sourceV1, sourceV2, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.zip.x2", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`intr_zip_x4`

"""
function intr_zip_x4(v1::Value, v2::Value, v3::Value, v4::Value; res::IR.Type, location=Location())
    results = IR.Type[res, ]
    operands = Value[v1, v2, v3, v4, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.intr.zip.x4", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`zip_x4`

This operation interleaves elements from four input SVE vectors, returning
four new SVE vectors, each of which contain a quarter of the result. The
first quarter will be in `resultV1`, second in `resultV2`, third in
`resultV3`, and fourth in `resultV4`.

```mlir
// sourceV1 = [ A1, A2, ... An ]
// sourceV2 = [ B1, B2, ... Bn ]
// sourceV3 = [ C1, C2, ... Cn ]
// sourceV4 = [ D1, D2, ... Dn ]
// (resultV1, resultV2, resultV3, resultV4)
//   = [ A1, B1, C1, D1, A2, B2, C2, D2, ... An, Bn, Cn, Dn ]
%resultV1, %resultV2, %resultV3, %resultV4 = arm_sve.zip.x4
  %sourceV1, %sourceV2, %sourceV3, %sourceV4 : vector<[16]xi8>
```

**Warning:** The result of this op is undefined for 64-bit elements on
hardware with less than 256-bit vectors!

Note: This requires SME 2 (`+sme2` in LLVM target features)

[Source](https://developer.arm.com/documentation/ddi0602/2023-12/SME-Instructions/ZIP--four-registers---Interleave-elements-from-four-vectors-?lang=en)
"""
function zip_x4(sourceV1::Value, sourceV2::Value, sourceV3::Value, sourceV4::Value; resultV1::IR.Type, resultV2::IR.Type, resultV3::IR.Type, resultV4::IR.Type, location=Location())
    results = IR.Type[resultV1, resultV2, resultV3, resultV4, ]
    operands = Value[sourceV1, sourceV2, sourceV3, sourceV4, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "arm_sve.zip.x4", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # arm_sve
