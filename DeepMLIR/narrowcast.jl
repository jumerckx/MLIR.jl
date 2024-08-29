### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 0bd3af91-e577-44ad-90f3-461e08500dfa
import Base.Broadcast: broadcasted, Broadcasted

# ╔═╡ 129f8aa6-654e-11ef-21f3-13bc6ce147b4
bc = broadcasted(sin, broadcasted(*, broadcasted(-, broadcasted(+, rand(10), rand(1, 9), 2), rand(10, 9)), rand(1, 9)))

# ╔═╡ 85bce120-d8a6-4fe7-a820-e48f0f8c54d9
struct Arg{N}; end

# ╔═╡ 86580b58-919b-473a-8565-aad5ed20f1bc
begin
	struct Eval{F, T}
		args::T
		
		function Eval{F}(args::T) where {F, T}
			new{F, T}(args)
		end
	end
	function Eval(bc::Broadcasted)
		first(Eval(bc, 1))
	end
	function Eval(bc::Broadcasted, argindex)
		args = []
		for arg in bc.args
			new_arg, argindex = Eval(arg, argindex)
			push!(args, new_arg)
		end
		return Eval{bc.f}(Tuple(args)), argindex
	end
	Eval(::Any, argindex) = Arg{argindex}(), argindex+1
end

# ╔═╡ eb68ffc2-dac0-4dfa-b069-1e6fc9bb0df1
e = Eval(bc)

# ╔═╡ 486ba3e1-0c23-4493-b06f-f6af7f6aa3c0
begin
	@inline (e::Eval{F})(argvalues...) where F = e(argvalues)
	
	@inline function (e::Eval{F})(argvalues::Tuple) where F
		return F(_eval_tuple(e.args, argvalues)...)
	end
	@inline (::Arg{N})(argvalues) where N = argvalues[N]
	
	@inline _eval_tuple(args::NTuple{N, Union{Eval, Arg}}, argvalues) where N = (first(args)(argvalues), _eval_tuple(Base.tail(args), argvalues)...)
	@inline _eval_tuple(::Tuple{}, argvalues) = ()
end

# ╔═╡ 2c300285-ee1e-474d-976e-19a01560f757
@code_typed e(1., 2, 3, 4, 5)

# ╔═╡ 85611acd-a9b2-4acb-a8aa-d509d2d2e0cb
@time e(1., 2, 3, 4, 5)

# ╔═╡ Cell order:
# ╠═0bd3af91-e577-44ad-90f3-461e08500dfa
# ╠═129f8aa6-654e-11ef-21f3-13bc6ce147b4
# ╠═85bce120-d8a6-4fe7-a820-e48f0f8c54d9
# ╠═86580b58-919b-473a-8565-aad5ed20f1bc
# ╠═eb68ffc2-dac0-4dfa-b069-1e6fc9bb0df1
# ╠═486ba3e1-0c23-4493-b06f-f6af7f6aa3c0
# ╠═2c300285-ee1e-474d-976e-19a01560f757
# ╠═85611acd-a9b2-4acb-a8aa-d509d2d2e0cb
