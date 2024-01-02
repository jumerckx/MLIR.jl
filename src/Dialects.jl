module Dialects

using CassetteOverlay
@MethodTable MLIRCompilation
mlircompilationpass = @overlaypass MLIRCompilation;

using MacroTools
import CassetteOverlay.nonoverlay

import ..IR: Attribute, NamedAttribute, context, Block
import ..API

namedattribute(name, val) = namedattribute(name, Attribute(val))
namedattribute(name, val::Attribute) = NamedAttribute(name, val)
function namedattribute(name, val::NamedAttribute)
    @assert true # TODO(jm): check whether name of attribute is correct, getting the name might need to be added to IR.jl?
    return val
end

operandsegmentsizes(segments) = namedattribute(
    "operand_segment_sizes",
    Attribute(API.mlirDenseI32ArrayGet(
        context().context,
        length(segments),
        Int32.(segments)
    )))


function nonoverlay(ex::Expr)
    @static if VERSION ≥ v"1.10.0-DEV.68"
        topmod = Core.Compiler._topmod(__module__)
        f, args, kwargs = Base.destructure_callex(topmod, ex)
    else
        f, args, kwargs = Base.destructure_callex(ex)
    end
    out = Expr(:call, GlobalRef(@__MODULE__, :nonoverlay))
    isempty(kwargs) || push!(out.args, Expr(:parameters, kwargs...))
    push!(out.args, f)
    append!(out.args, args)
    return out
end

# from https://github.com/JuliaLang/julia/blob/1b183b93f4b78f567241b1e7511138798cea6a0d/base/experimental.jl#L345C1-L357C4
function overlay_def!(mt, @nospecialize ex)
    arg1 = ex.args[1]
    if isexpr(arg1, :call)
        arg1.args[1] = Expr(:overlay, mt, arg1.args[1])
    elseif isexpr(arg1, :(::))
        overlay_def!(mt, arg1)
    elseif isexpr(arg1, :where)
        overlay_def!(mt, arg1)
    else
        error("@overlay requires a function definition")
    end
    return ex
end

function mlirop_(expr)
    dict = splitdef(expr)
    rtype = get(dict, :rtype, :Any)

    modified = :($(dict[:name])($(dict[:args]...); $(dict[:kwargs]...)))
    modified = nonoverlay(modified)
    @info modified
    modified = :(
        function $(dict[:name])($(dict[:args]...); $(dict[:kwargs]...))::$rtype where {$(dict[:whereparams]...)}
            $modified
            println("overlayed!!!!")
            push!(currentblock(), op)
        end
    )

    modified = overlay_def!(:MLIRCompilation, modified)
    return quote
        $(expr)

        $modified
    end |> esc
end

macro mlirop(f)
    f = macroexpand(__module__, f)
    Base.is_function_def(f) || error("@mlirop requires a function definition")
    mlirop_(f)
end

include("dialects/builtin.jl")

include("dialects/llvm.jl")

include("dialects/arith.jl")

include("dialects/cf.jl")

include("dialects/func.jl")

# include("dialects/Gpu.jl")

# include("dialects/Memref.jl")

# include("dialects/Index.jl")

include("dialects/affine.jl")

# include("dialects/Ub.jl")

# include("dialects/SCF.jl")

end # module Dialects
