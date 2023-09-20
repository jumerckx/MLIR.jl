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
#include "mlir/TableGen/Region.h"
#include "mlir/TableGen/SideEffects.h"
#include "mlir/TableGen/Trait.h"

namespace {

llvm::cl::opt<bool> ExplainMissing(
    "explain-missing",
    llvm::cl::desc("Print the reason for skipping operations from output"));

llvm::cl::opt<std::string> StripOpPrefix(
    "strip-prefix", llvm::cl::desc("Prefix to strip from def names"),
    llvm::cl::value_desc("prefix"));

llvm::cl::opt<std::string> DialectName(
    "dialect-name", llvm::cl::desc("Override the inferred dialect name"),
    llvm::cl::value_desc("dialect"));

template <class C>
llvm::iterator_range<typename C::const_iterator> make_range(const C& x) {
  return llvm::make_range(x.begin(), x.end());
}

template <class C, class FunTy,
          typename ResultTy = decltype(std::declval<FunTy>()(
              std::declval<typename C::value_type>()))>
std::vector<ResultTy> map_vector(const C& container, FunTy f) {
  std::vector<ResultTy> results;
  for (const auto& v : container) {
    results.push_back(f(v));
  }
  return results;
}

void warn(llvm::StringRef op_name, const std::string& reason) {
  if (!ExplainMissing) return;
  llvm::errs() << llvm::formatv(
      "{0} {1}\n", llvm::fmt_align(op_name, llvm::AlignStyle::Left, 40),
      reason);
}

void warn(const mlir::tblgen::Operator& op, const std::string& reason) {
  warn(op.getOperationName(), reason);
}

struct AttrPatternTemplate {
  const char* _pattern;
  const char* _type;
  std::vector<const char*> provided_constraints;
  std::vector<const char*> type_var_defaults;
};

using attr_print_state = llvm::StringSet<>;
class AttrPattern {
 public:
  virtual ~AttrPattern() = default;
  virtual std::string type() const = 0;
  virtual std::string match(std::string name) const = 0;
  virtual const std::vector<std::string>& provided_constraints() const = 0;
  virtual void print(llvm::raw_ostream& os,
                     attr_print_state& optional_attr_defs) const = 0;
};

struct NameSource {
  NameSource(const char* prefix) : prefix(prefix) {}
  NameSource(const NameSource&) = delete;
  std::string fresh() { return std::string(prefix) + std::to_string(suffix++); }
 private:
    const char* prefix;
    int suffix = 0;
};

class SimpleAttrPattern : public AttrPattern {
 public:
  SimpleAttrPattern(const AttrPatternTemplate& tmpl, NameSource& gen)
    : _type_var_defaults(tmpl.type_var_defaults) {
    _pattern = tmpl._pattern;
    if (tmpl.type_var_defaults.empty()) {
      _type = tmpl._type;
      _provided_constraints =
          map_vector(tmpl.provided_constraints,
                     [](const char* c) { return std::string(c); });
    } else if (tmpl.type_var_defaults.size() == 1) {
      std::string var = gen.fresh();
      _type_vars.push_back(var);
      _type = llvm::formatv(tmpl._type, var);
      _provided_constraints = map_vector(
          tmpl.provided_constraints,
          [&var](const char* c) { return llvm::formatv(c, var).str(); });
    } else {
      std::abort();  // Not sure how to splat arbitrary many vars into formatv.
    }
  }

  std::string match(std::string name) const override { return llvm::formatv(_pattern, name); }
  std::string type() const override { return "Union{NamedAttribute, " + _type + "}"; }
  const std::vector<std::string>& provided_constraints() const override { return _provided_constraints; }
  const std::vector<std::string>& type_vars() const { return _type_vars; }
  const std::vector<const char*>& type_var_defaults() const { return _type_var_defaults; }

  void print(llvm::raw_ostream& os,
             attr_print_state& optional_attr_defs) const override {}
 private:
  const char* _pattern;
  std::string _type;
  std::vector<std::string> _provided_constraints;
  std::vector<std::string> _type_vars;
  const std::vector<const char*> _type_var_defaults;
};

class OptionalAttrPattern : public AttrPattern {
 public:
  OptionalAttrPattern(llvm::StringRef attr_kind, SimpleAttrPattern base)
    : base(std::move(base)), attr_kind(attr_kind) {}

  std::string type() const override {
    return "Union{Nothing, " + base.type() + "}";
  }
  std::string match(std::string name) const override {
    return llvm::formatv("Optional{0} {1}", attr_kind, name);
  }
  const std::vector<std::string>& provided_constraints() const override { return base.provided_constraints(); }

  void print(llvm::raw_ostream& os,
             attr_print_state& optional_attr_defs) const override {
  }

 private:
  SimpleAttrPattern base;
  llvm::StringRef attr_kind;
};

using attr_pattern_map = llvm::StringMap<AttrPatternTemplate>;

const attr_pattern_map& getAttrPatternTemplates() {
  static const attr_pattern_map* kAttrHandlers = new attr_pattern_map{
    {"BoolAttr", {"", "Bool", {}, {}}},
    {"DenseI32ArrayAttr", {"", "Vector{Float32}", {}, {}}},
    {"F32Attr", {"", "Float32", {}, {}}},
    {"F64Attr", {"", "Float64", {}, {}}},
    {"I32Attr", {"", "Int32", {}, {}}},
    {"I64Attr", {"", "Int64", {}, {}}},
    {"I64ArrayAttr", {"", "Vector{Int64}", {}, {}}},
    {"StrAttr", {"", "String", {}, {}}},
      // {"AnyAttr", {"{0}", "Attribute", {}, {}}},
      // {"AffineMapArrayAttr", {"PatternUtil.AffineMapArrayAttr {0}", "[Affine.Map]", {}, {}}},
      // {"AffineMapAttr", {"AffineMapAttr {0}", "Affine.Map", {}, {}}},
      // {"ArrayAttr", {"ArrayAttr {0}", "[Attribute]", {}, {}}},
      // {"BoolAttr", {"BoolAttr {0}", "Bool", {}, {}}},
      // {"DenseI32ArrayAttr", {"PatternUtil.I32ArrayAttr {0}", "[Int]", {}, {}}},
      // {"DictionaryAttr", {"DictionaryAttr {0}", "(M.Map Name Attribute)", {}, {}}},
      // {"F32Attr", {"FloatAttr Float32Type {0}", "Double", {}, {}}},
      // {"F64Attr", {"FloatAttr Float64Type {0}", "Double", {}, {}}},
      // {"I32Attr", {"IntegerAttr (IntegerType Signless 32) {0}", "Int", {}, {}}},
      // {"I64Attr", {"IntegerAttr (IntegerType Signless 64) {0}", "Int", {}, {}}},
      // {"I64ArrayAttr", {"PatternUtil.I64ArrayAttr {0}", "[Int]", {}, {}}},
      // {"I64ElementsAttr", {"DenseElementsAttr (IntegerType Signless 64) (DenseInt64 {0})",
      //                      "(AST.IStorableArray {0} Int64)", {"Ix {0}", "Show {0}"}, {"PatternUtil.DummyIx"}}},
      // {"IndexAttr", {"IntegerAttr IndexType {0}", "Int", {}, {}}},
      // {"StrAttr", {"StringAttr {0}", "BS.ByteString", {}, {}}},
      // // TODO(jpienaar): We could specialize this one more to query Type.
      // {"TypedAttrInterface", {"{0}", "Attribute", {}, {}}},
  };
  return *kAttrHandlers;
}

// Returns nullptr when the attribute pattern couldn't be constructed.
std::unique_ptr<AttrPattern> tryGetAttrPattern(
    const mlir::tblgen::NamedAttribute& nattr, NameSource& gen) {
  llvm::StringRef attr_kind = nattr.attr.getAttrDefName();
  const AttrPatternTemplate* tmpl; // Declare tmpl as a pointer

  if (getAttrPatternTemplates().count(attr_kind) != 1) {
      static const AttrPatternTemplate defaultTemplate = {"", "Attribute", {}, {}};
      tmpl = &defaultTemplate;
  } else {
      const AttrPatternTemplate& attrTemplate = getAttrPatternTemplates().lookup(attr_kind);
      tmpl = &attrTemplate;
  }

  // if (getAttrPatternTemplates().count(attr_kind) != 1) return nullptr;
  // const AttrPatternTemplate& tmpl = getAttrPatternTemplates().lookup(attr_kind);
  if (!nattr.attr.isOptional()) {
    return std::make_unique<SimpleAttrPattern>(*tmpl, gen);
  } else {
    auto pat = std::make_unique<OptionalAttrPattern>(
        attr_kind, SimpleAttrPattern(*tmpl, gen));
    return pat;
  }
}

const std::string sanitizeName(llvm::StringRef name, std::optional<int> idx = std::nullopt) {
  static const llvm::StringSet<>* kReservedNames = new llvm::StringSet<>{
      // TODO(apaszke): Add more keywords
      // Haskell keywords
      // "in", "data", "if"
  };
  if (name.empty()) {
    assert(idx);
    return llvm::formatv("_unnamed{0}", *idx);
  } else if (kReservedNames->contains(name)) {
    auto new_name = name.str();
    new_name.push_back('_');
    return new_name;
  } else {
    return name.str();
  }
}

std::string getDialectName(llvm::ArrayRef<llvm::Record*> op_defs) {
  mlir::tblgen::Operator any_op(op_defs.front());
  assert(
      std::all_of(op_defs.begin(), op_defs.end(), [&any_op](llvm::Record* op) {
        return mlir::tblgen::Operator(op).getDialectName() ==
               any_op.getDialectName();
      }));
  std::string dialect_name;
  if (DialectName.empty()) {
    dialect_name = any_op.getDialectName().str();
    dialect_name[0] = llvm::toUpper(dialect_name[0]);
  } else {
    dialect_name = DialectName;
  }
  return dialect_name;
}

class OpAttrPattern {
  OpAttrPattern(std::string name, std::vector<std::string> binders,
                std::vector<std::string> default_values,
                std::vector<mlir::tblgen::NamedAttribute> attrs,
                std::vector<std::unique_ptr<AttrPattern>> patterns)
      : name(std::move(name)),
        binders(std::move(binders)),
        default_values(std::move(default_values)),
        attrs(std::move(attrs)),
        patterns(std::move(patterns)) {}

 public:
  static std::optional<OpAttrPattern> buildFor(mlir::tblgen::Operator& op) {
    if (op.getNumAttributes() == 0) return OpAttrPattern("NoAttrs", {}, {}, {}, {});

    NameSource gen("a");
    std::vector<std::string> binders;
    std::vector<std::string> default_values;
    std::vector<mlir::tblgen::NamedAttribute> attrs;
    std::vector<std::unique_ptr<AttrPattern>> patterns;
    for (const auto& named_attr : op.getAttributes()) {
      // Derived attributes are never materialized and don't have to be
      // specified.
      if (named_attr.attr.isDerivedAttr()) continue;

      auto pattern = tryGetAttrPattern(named_attr, gen);
      binders.push_back(sanitizeName(named_attr.name) + "_");

      if (named_attr.attr.isOptional()) {
        default_values.push_back("=nothing");
      }
      else {
        default_values.push_back("");
      }

      attrs.push_back(named_attr);
      patterns.push_back(std::move(pattern));
    }
    if (binders.empty()) return OpAttrPattern("NoAttrs", {}, {}, {}, {});
    std::string name = "Internal" + op.getCppClassName().str() + "Attributes";
    return OpAttrPattern(std::move(name), std::move(binders), std::move(default_values),
                        std::move(attrs), std::move(patterns));
  }

  void print(llvm::raw_ostream& os, attr_print_state& optional_attr_defs) const {
    std::vector<std::string> required_attr_creator;

    std::vector<std::string> optional_attr_creator;

    for (size_t i = 0; i < attrs.size(); ++i) {
      const mlir::tblgen::NamedAttribute& nattr = attrs[i];
      const AttrPattern& pattern = *patterns[i];
      std::string inst_pattern = pattern.match(binders[i]);
      if (nattr.attr.isOptional()) {
        optional_attr_creator.push_back(llvm::formatv("({1} != nothing) && push!(attributes, make_named_attribute(\"{0}\", {1}))", nattr.name, binders[i]));
      } else {
        required_attr_creator.push_back(llvm::formatv("make_named_attribute(\"{0}\", {1})", nattr.name, binders[i]));
      }
    }
    const char* kAttributePattern = R"(attributes = [{0:$[, ]}]{2}
  {1:$[
  ]
  }{2})";
    os << llvm::formatv(kAttributePattern,
                        make_range(required_attr_creator),
                        make_range(optional_attr_creator),
                        (optional_attr_creator.empty() ? "" : "\n"));
}

  std::vector<std::string> types() const {
    return map_vector(patterns, [](const std::unique_ptr<AttrPattern>& p) {
      return p->type();
    });
  }
  std::vector<std::string> provided_constraints() const {
    std::vector<std::string> result;
    for (auto& p : patterns) {
      for (auto& c : p->provided_constraints()) {
        result.push_back(c);
      }
    }
    return result;
  }

  std::string name;
  std::vector<std::string> binders;
  std::vector<std::string> default_values;

 private:
  std::vector<mlir::tblgen::NamedAttribute> attrs;
  std::vector<std::unique_ptr<AttrPattern>> patterns;
};

std::optional<std::string> buildOperation(
    const llvm::Record* def, bool is_pattern, const std::string& what_for,
    const std::string& location_expr,
    const std::vector<std::string>& type_exprs,
    const std::vector<std::string>& operand_exprs,
    const std::vector<std::string>& region_exprs,
    const OpAttrPattern& attr_pattern) {
  mlir::tblgen::Operator op(def);
  auto fail = [&op, &what_for](std::string reason) {
    warn(op, llvm::formatv("couldn't construct {0}: {1}", what_for, reason));
    return std::optional<std::string>();
  };

  // Skip currently unsupported cases
  if (op.getNumVariadicRegions() != 0) return fail("variadic regions");
  if (op.getNumSuccessors() != 0) return fail("successors");

  // Prepare results
  std::string type_expr;
  if (op.getNumResults() == 0) {
    assert(type_exprs.size() == op.getNumResults());
    type_expr = "[]";
  } else if (op.getNumVariableLengthResults() == 0 &&
             op.getTrait("::mlir::OpTrait::SameOperandsAndResultType")) {
    assert(type_exprs.size() == 1);
    type_expr = llvm::formatv("[{0:$[, ]}]",
                              make_range(std::vector<llvm::StringRef>(
                                  op.getNumResults(), type_exprs.front())));
  } else if (op.getNumVariableLengthResults() == 0) {
    assert(type_exprs.size() == op.getNumResults());
    type_expr = llvm::formatv("[{0:$[, ]}]", make_range(type_exprs));
  } else if (!is_pattern) {
    // TODO(jumerckx): when does this happen?
    assert(type_exprs.size() == op.getNumResults());
    std::vector<std::string> list_type_exprs;
    for (int i = 0; i < op.getNumResults(); ++i) {
      auto& result = op.getResult(i);
      if (result.isOptional()) {
        list_type_exprs.push_back("(Data.Maybe.maybeToList " + type_exprs[i] + ")");
      } else if (result.isVariadic()) {
        list_type_exprs.push_back(type_exprs[i]);
      } else {
        assert(!result.isVariableLength());
        list_type_exprs.push_back("[" + type_exprs[i] + "]");
      }
    }
    type_expr = llvm::formatv("({0:$[ ++ ]})", make_range(list_type_exprs));
  } else {
    return fail("unsupported variable length results");
  }

  // Prepare operands
  std::string operand_expr;
  assert(operand_exprs.size() == op.getNumOperands());
  if (op.getNumOperands() == 1 && op.getOperand(0).isVariadic()) {
    //TODO(jumerckx): should probably be operand_expr="[]"
    // Note that this expr already should represent a list
    operand_expr = operand_exprs.front();
  } else if (op.getNumVariableLengthOperands() == 0) {
    operand_expr = llvm::formatv("[{0:$[, ]}]", make_range(operand_exprs));
  } else if (!is_pattern) {
    // TODO(jumerckx): when does this happen?
    std::vector<std::string> operand_list_exprs;
    for (int i = 0; i < op.getNumOperands(); ++i) {
      auto& operand = op.getOperand(i);
      if (operand.isOptional()) {
        operand_list_exprs.push_back("(Data.Maybe.maybeToList " + operand_exprs[i] + ")");
      } else if (operand.isVariadic()) {
        operand_list_exprs.push_back(operand_exprs[i]);
      } else {
        assert(!operand.isVariableLength());
        operand_list_exprs.push_back("[" + operand_exprs[i] + "]");
      }
    }
    operand_expr =
        llvm::formatv("({0:$[ ++ ]})", make_range(operand_list_exprs));
  } else {
    return fail("unsupported variable length operands");
  }

  std::string extra_attrs;
  if (op.getTrait("::mlir::OpTrait::AttrSizedOperandSegments")) {
    std::vector<std::string> segment_sizes;
    for (int i = 0; i < op.getNumOperands(); ++i) {
      auto& operand = op.getOperand(i);
      if (operand.isOptional()) {
        segment_sizes.push_back(llvm::formatv(
            "case {0} of Just _ -> 1; Nothing -> 0", operand_exprs[i]));
      } else if (operand.isVariadic()) {
        segment_sizes.push_back("Prelude.length " + operand_exprs[i]);
      } else {
        assert(!operand.isVariableLength());
        segment_sizes.push_back("1");
      }
    }
  }

  const char* kPatternExplicitType = R"(create_operation(
        "{0}", {1}, 
        results = {2}, 
        operands = {3},
        owned_regions = [{4:$[, ]}], 
        successors = [], 
        attributes = attributes,
        result_inference=false
      ))";
  return llvm::formatv(kPatternExplicitType,
                       op.getOperationName(),                    // 0
                       location_expr,                            // 1
                       type_expr,                                // 2
                       operand_expr,                             // 3
                       make_range(region_exprs),                 // 4
                       attr_pattern.name,                        // 5
                       make_range(attr_pattern.binders))        // 6
      .str();
}

// TODO(apaszke): Make this more reliable
std::string legalizeBuilderName(std::string name) {
  for (size_t i = 0; i < name.size(); ++i) {
    if (name[i] == '.') name[i] = '_';
  }
  return name;
}

std::string stripDialect(std::string name) {
  size_t dialect_sep_loc = name.find('.');
  assert(dialect_sep_loc != std::string::npos);
  return name.substr(dialect_sep_loc + 1);
}

void emitPattern(const llvm::Record* def, const OpAttrPattern& attr_pattern,
                 llvm::raw_ostream& os) {
  mlir::tblgen::Operator op(def);
  auto fail = [&op](std::string reason) {
    return warn(op, llvm::formatv("couldn't construct pattern: {0}", reason));
  };

  // Skip currently unsupported cases
  if (op.getNumVariableLengthResults() != 0) return fail("variadic results");
  if (op.getNumRegions() != 0) return fail("regions");
  if (op.getNumSuccessors() != 0) return fail("successors");
  if (!def->getName().endswith("Op")) return fail("unsupported name format");
  if (!def->getName().startswith(StripOpPrefix)) return fail("missing prefix");

  // Drop the stripped prefix and "Op" from the end.
  llvm::StringRef pattern_name =
      def->getName().drop_back(2).drop_front(StripOpPrefix.length());

  std::vector<std::string> pattern_arg_types{"Location"};

  // Prepare results
  std::vector<std::string> result_binders;
  if (op.getNumResults() > 0 &&
      op.getTrait("::mlir::OpTrait::SameOperandsAndResultType")) {
    assert(op.getNumVariableLengthResults() == 0);
    pattern_arg_types.push_back("MLIRType");
    result_binders.push_back("type");
  } else {
    size_t result_count = 0;
    for (int i = 0; i < op.getNumResults(); ++i) {
      pattern_arg_types.push_back("MLIRType");
      result_binders.push_back(llvm::formatv("type_{0}", result_count++));
    }
  }

  // Prepare operands
  std::vector<std::string> operand_binders;
  if (op.getNumOperands() == 1 && op.getOperand(0).isVariadic()) {
    // Single variadic arg is easy to handle
    pattern_arg_types.push_back("Vector{Value}");
    operand_binders.push_back(sanitizeName(op.getOperand(0).name, 0) + "_");
  } else {
    // Non-variadic case
    for (int i = 0; i < op.getNumOperands(); ++i) {
      const auto& operand = op.getOperand(i);
      if (operand.isVariableLength())
        return fail("unsupported variable length operand");
      pattern_arg_types.push_back("Value");
      operand_binders.push_back(sanitizeName(operand.name, i) + "_");
    }
  }

  // Prepare attribute pattern
  auto attr_types = attr_pattern.types();
  pattern_arg_types.insert(pattern_arg_types.end(), attr_types.begin(),
                           attr_types.end());

  std::optional<std::string> operation = buildOperation(
      def, true, "pattern", "location",
      result_binders, operand_binders, {}, attr_pattern);
  if (!operation) return;

  std::vector<std::string> binders;
  std::vector<std::string> default_values;

  binders.push_back("location");
  binders.insert(binders.end(), result_binders.begin(), result_binders.end());
  binders.insert(binders.end(), operand_binders.begin(), operand_binders.end());
  binders.insert(binders.end(), attr_pattern.binders.begin(), attr_pattern.binders.end());

  // fill default values with empty strings except for attributes
  default_values.push_back("");
  default_values.insert(default_values.end(), result_binders.size(), "");
  default_values.insert(default_values.end(), operand_binders.size(), "");
  default_values.insert(default_values.end(), attr_pattern.default_values.begin(), attr_pattern.default_values.end());

  std::vector<std::string> all_args;
  for (size_t i = 0; i < binders.size(); ++i) {
    all_args.push_back(llvm::formatv("{0}{1}::{2}", binders[i], default_values[i], pattern_arg_types[i]));
  }
  
  std::string attribute_definitions;
  llvm::raw_string_ostream stream(attribute_definitions);
  attr_print_state attr_pattern_state;
  attr_pattern.print(stream, attr_pattern_state);

  const char* kPatternExplicitType = R"(
function {0}({1:$[, ]})
  {3}
  {2}
end
)";
  os << llvm::formatv(kPatternExplicitType,
                      pattern_name,                                      // 0
                      make_range(all_args),                           // 1
                      *operation,                                        // 2
                      attribute_definitions);                            // 3

}

std::string formatDescription(mlir::tblgen::Operator op) {
  std::string description;
  description = op.getDescription().str();
  size_t pos = 0;
  while (description[pos] == '\n') ++pos;
  size_t leading_spaces = 0;
  while (description[pos++] == ' ') ++leading_spaces;
  if (leading_spaces) {
    std::string leading_spaces_str;
    for (size_t i = 0; i < leading_spaces; ++i) leading_spaces_str += "[ ]";
    description = std::regex_replace(description, std::regex("\n" + leading_spaces_str), "\n");
  }
  description = std::regex_replace(description, std::regex("(['\"$#])"), "\\$1");
  return description;
}

}  // namespace


bool emitOpTableDefs(const llvm::RecordKeeper& recordKeeper,
                     llvm::raw_ostream& os) {
  std::vector<llvm::Record*> defs = recordKeeper.getAllDerivedDefinitions("Op");

  if (defs.empty()) return true;
  // TODO(apaszke): Emit a module header to avoid leaking internal definitions.
  auto dialect_name = getDialectName(defs);
  os << "module " << dialect_name << "\n";
  os << R"(
make_named_attribute(name, val) = make_named_attribute(name, Attribute(val))

make_named_attribute(name, val::Attribute) = NamedAttribute(name, val)

function make_named_attribute(name, val::NamedAttribute)
  assert(true) # TODO(jm): check whether name of attribute is correct, getting the name might need to be added to IR.jl?
  return val
end
)";

  attr_print_state attr_pattern_state;
  for (const auto* def : defs) {
    os << "\n";
    mlir::tblgen::Operator op(*def);
    if (op.hasDescription()) {
      os << "\"\"\"\n" << stripDialect(op.getOperationName()) << "\n";
      os << formatDescription(op);
      os << "\n\"\"\"";
    }
    std::optional<OpAttrPattern> attr_pattern = OpAttrPattern::buildFor(op);
    if (!attr_pattern) continue;
    emitPattern(def, *attr_pattern, os);
    os << "\n";
  }

  os << "\nend #" << dialect_name << "\n";

  return false;
}
