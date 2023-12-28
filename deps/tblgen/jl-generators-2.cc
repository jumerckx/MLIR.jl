// Copyright 2021 Google LLC
// Copyright 2023 Valentin Churavy
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <functional>
#include <regex>
#include <optional>
#include <iostream>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatCommon.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "mlir/TableGen/Argument.h"
#include "mlir/TableGen/Class.h"
#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Region.h"
#include "mlir/TableGen/SideEffects.h"
#include "mlir/TableGen/Trait.h"

namespace
{
    using namespace mlir;
    using namespace mlir::tblgen;

    /// Returns true if the SameArgumentAndResultTypes trait can be used to infer
    /// result types of the given operation.
    static bool hasSameArgumentAndResultTypes(const Operator &op)
    {
        return op.getTrait("::mlir::OpTrait::SameOperandsAndResultType") &&
               op.getNumVariableLengthResults() == 0;
    }

    /// Returns true if the FirstAttrDerivedResultType trait can be used to infer
    /// result types of the given operation.
    static bool hasFirstAttrDerivedResultTypes(const Operator &op)
    {
        return op.getTrait("::mlir::OpTrait::FirstAttrDerivedResultType") &&
               op.getNumVariableLengthResults() == 0;
    }

    /// Returns true if the InferTypeOpInterface can be used to infer result types
    /// of the given operation.
    static bool hasInferTypeInterface(const Operator &op)
    {
        return op.getTrait("::mlir::InferTypeOpInterface::Trait") &&
               op.getNumRegions() == 0;
    }

    /// Returns true if there is a trait or interface that can be used to infer
    /// result types of the given operation.
    static bool canInferType(const Operator &op)
    {
        return hasSameArgumentAndResultTypes(op) ||
               hasFirstAttrDerivedResultTypes(op) || hasInferTypeInterface(op);
    }
}

bool emitOpTableDefs(const llvm::RecordKeeper &recordKeeper,
                     llvm::raw_ostream &os)
{

    // std::vector<llvm::Record *> types = recordKeeper.getAllDerivedDefinitionsIfDefined("Type");
    // os << "\n---Types---\n";
    // for (const auto *def : types)
    // {
    //     mlir::tblgen::Type type(def);

    //     os << type.getDefName() << "\n";
    // }

    std::vector<llvm::Record *> typedefs = recordKeeper.getAllDerivedDefinitionsIfDefined("TypeDef");
    os << "\n---TypeDefs---\n";
    const char *typetemplate = R"(struct {0} <: AbstractValue
    value::MlirValue
    function {0}(v::AbstractValue)
        get_type(v) == parse(MLIRType, "{1}") || error("Expected {0}, got ", get_type(v))
        return new(v.value)
    end
end
)";

    for (const auto *def : typedefs)
    {
        mlir::tblgen::TypeDef type(def);

        if (type.getNumParameters() > 0)
        {
            // parametric types are not supported yet
            break;
        }
        auto mnemonic = type.getMnemonic();
        if (!mnemonic.has_value())
        {
            break;
        }

        auto dialectname = type.getDialect().getName();
        std::string parseableformat = "!" + dialectname.str() + "<" + mnemonic.value().str() + ">";

        std::string name = mnemonic.value().str();

        if (mnemonic.value().equals_insensitive(dialectname))
        {
            name += "Type";
        }

        name[0] = std::toupper(name[0]);

        os << llvm::formatv(typetemplate, name, parseableformat);

        // os << type.getName() << ": "
        //    << "!" << dialectname << "<" << type.getMnemonic() << ">"
        //    << ", " << type.getAssemblyFormat() << "\n";

        // for (size_t i = 0; i < type.getNumParameters(); i++)
        // {
        //     const auto &named_param = type.getParameters()[i];
        //     os << "\t" << named_param.getName() << " (parameter)\n";
        // }
    }

    os << "\n---Attributes---\n";
    std::vector<llvm::Record *> attributes = recordKeeper.getAllDerivedDefinitionsIfDefined("AttrDef");
    for (const auto *def : attributes)
    {
        mlir::tblgen::AttrDef attrdef(def);

        os << attrdef.getName() << "\n";
    }

    os << "\n---Operations---\n";
    std::vector<llvm::Record *> opdefs = recordKeeper.getAllDerivedDefinitionsIfDefined("Op");

    const char *functiontemplate = R"(
function {0}({1}location=Location())
    {2}
end
)";      // 0: functionname, 1: functionarguments, 2: functionbody
    const char *functionbodytemplate = R"(results = [{0}]
    operands = [{1}]
    owned_regions = [{2}]
    successors = [{3}]
    attributes = [{4}]

    {5}

    create_operation(
        {6}, location,
        results=results,
        operands=operands,
        owned_regions=regions,
        successors=successors,
        attributes=attributes,
        result_inference={7}
    ))"; // 0: results, 1: operands, 2: owned_regions, 3: successors, 4: attributes, 5: optionals, 6: opname, 7: result_inference

    for (const auto *def : opdefs)
    {
        mlir::tblgen::Operator op(*def);

        std::string operandarguments = "";
        std::string operandcontainer = "";
        std::string optionals = "";

        auto opname = op.getOperationName();

        bool inferrable = canInferType(op);

        for (size_t i = 0; i < op.getNumOperands(); i++)
        {
            const auto &named_operand = op.getOperand(i);
            std::string defaultvalue = "";
            std::string operandname = named_operand.name.str();
            if (operandname.empty())
            {
                operandname = "operand_" + std::to_string(i);
            }

            // auto type = named_operand.constraint.getPredicate().getCondition();
            std::string type = "Value";

            bool optional = named_operand.isOptional();
            bool variadic = named_operand.isVariadic();

            if (variadic)
            {
                type = "Vector{" + type + "}";
            }

            if (optional)
            {
                optionals += llvm::formatv(R"(({0} != nothing) && push!(operands, {0}{1})
    )",
                                           operandname, (variadic ? "..." : ""));
                type = "Union{Nothing, " + type + "}";
                defaultvalue = "=nothing";
            }
            else
            {
                operandcontainer += operandname + (variadic ? "..." : "") + ", ";
            }
            operandarguments += operandname + defaultvalue + "::" + type + (i == op.getNumOperands() - 1 ? "; " : ", ");
        }

        if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments"))
        {
            std::string operandsegmentsizes = "";
            for (size_t i = 0; i < op.getNumOperands(); i++)
            {
                const auto &named_operand = op.getOperand(i);
                std::string operandname = named_operand.name.str();
                if (operandname.empty())
                {
                    operandname = "operand_" + std::to_string(i);
                }
                if (named_operand.isOptional())
                {
                    operandsegmentsizes += "(" + operandname + "==nothing) ? 0 : 1";
                    continue;
                }
                operandsegmentsizes += named_operand.isVariadic() ? "length(" + operandname + "), " : "1, ";
            }
            optionals += llvm::formatv(R"(push!(attributes, operandsegmentsizes([{0}]))
    )",
                                       operandsegmentsizes);
        }

        std::string resultarguments = "";
        std::string resultcontainer = "";
        for (size_t i = 0; i < op.getNumResults(); i++)
        {
            const auto &named_result = op.getResult(i);
            std::string defaultvalue = "";
            std::string resultname = named_result.name.str();
            if (resultname.empty())
            {
                resultname = "result_" + std::to_string(i);
            }
            std::string type = "MLIRType";

            bool optional = named_result.isOptional() || inferrable;
            bool variadic = named_result.isVariadic();

            if (variadic)
            {
                type = "Vector{" + type + "}";
            }

            if (optional)
            {
                optionals += llvm::formatv(R"(({0} != nothing) && push!(results, {0}{1})
    )",
                                           resultname, (variadic ? "..." : ""));
                type = "Union{Nothing, " + type + "}";
                defaultvalue = "=nothing";
            }
            else
            {
                resultcontainer += resultname + (variadic ? "..." : "") + ", ";
            }
            resultarguments += resultname + defaultvalue + "::" + type + ", ";
        }

        std::string resultinference = (inferrable ? "(length(results) == 0 ? true : false)" : "false");

        std::string attributearguments = "";
        std::string attributecontainer = "";
        for (size_t i = 0; i < op.getNumAttributes(); i++)
        {
            const auto &named_attr = op.getAttribute(i);

            // Derived attributes are never materialized and don't have to be
            // specified.
            if (named_attr.attr.isDerivedAttr())
                continue;

            std::string defaultvalue = "";
            std::string attributename = named_attr.name.str();
            if (attributename.empty())
            {
                attributename = "attribute_" + std::to_string(i);
            }
            std::string type = "Union{Attribute, NamedAttribute}";

            bool optional = named_attr.attr.isOptional() || named_attr.attr.hasDefaultValue();

            if (optional)
            {
                optionals += llvm::formatv(R"(({0} != nothing) && push!(attributes, namedattribute("{0}", {0}))
    )",
                                           attributename);
                type = "Union{Nothing, " + type + "}";
                defaultvalue = "=nothing";
            }
            else
            {
                attributecontainer += "namedattribute(\"" + attributename + "\", " + attributename + "), ";
            }
            attributearguments += attributename + defaultvalue + "::" + type + ", ";
        }

        std::string regionarguments = "";
        std::string regioncontainer = "";
        for (size_t i = 0; i < op.getNumRegions(); i++)
        {
            const auto &named_region = op.getRegion(i);
            std::string defaultvalue = "";
            std::string regionname = named_region.name.str();
            if (regionname.empty())
            {
                regionname = "region_" + std::to_string(i);
            }
            std::string type = "Region";

            bool variadic = named_region.isVariadic();

            if (variadic)
            {
                type = "Vector{" + type + "}";
            }

            regioncontainer += regionname + (variadic ? "..." : "") + ", ";
            regionarguments += regionname + defaultvalue + "::" + type + ", ";
        }

        std::string successorarguments = "";
        std::string successorcontainer = "";
        for (size_t i = 0; i < op.getNumSuccessors(); i++)
        {
            const auto &named_successor = op.getSuccessor(i);
            std::string defaultvalue = "";
            std::string successorname = named_successor.name.str();
            if (successorname.empty())
            {
                successorname = "successor_" + std::to_string(i);
            }
            std::string type = "Block";

            bool variadic = named_successor.isVariadic();
            if (variadic)
            {
                type = "Vector{" + type + "}";
            }

            successorcontainer += successorname + (variadic ? "..." : "") + ", ";
            successorarguments += successorname + defaultvalue + "::" + type + ", ";
        }

        std::string arguments = operandarguments + resultarguments + attributearguments + regionarguments + successorarguments;

        std::string functionbody = llvm::formatv(functionbodytemplate, resultcontainer, operandcontainer, regioncontainer, successorcontainer, attributecontainer, optionals, opname, resultinference);
        os << llvm::formatv(functiontemplate, opname, arguments, functionbody);
    }
    return false;
}
