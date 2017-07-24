export mapreduce, mapfoldl, mapfoldr, reduce, foldl, foldr, sum, prod, mapreducedim

import Nabla.DiffBase.fmad

# Intercepts for `mapreduce`, `mapfoldl` and `mapfoldr` under `op` `+`.
accepted_add = :(Tuple{Any, typeof(+), AbstractArray{<:Real}})
for (f_name, base_f_name) in ((:mapreduce, :(Base.mapreduce)),
                              (:mapfoldl, :(Base.mapfoldl)),
                              (:mapfoldr, :(Base.mapfoldr)))
    eval(DiffBase, add_intercept(f_name, base_f_name, accepted_add))
    f_eval = eval(DiffBase, f_name)
    @eval DiffBase function ∇(
        ::typeof($f_eval),
        ::Type{Arg{3}},
        p, y, ȳ, f,
        ::typeof(+),
        A::AbstractArray{<:Real},
    )
        if needs_output(f) && method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, Real, Any})
            return Base.broadcast(An->ȳ * ∇(f, Arg{1}, An, f(An)), A)
        elseif !needs_output(f) && method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, Real})
            return Base.broadcast(An->ȳ * ∇(f, Arg{1}, An), A)
        else
            return Base.broadcast(An->ȳ * fmad(f, (An,), Val{1}), A)
        end
    end
end

# Implementation of `foldl`, `foldr` and `reduce` to avoid going into `Base`.
# Copied directly from `Base`.
foldl(op, v0, itr) = mapfoldl(identity, op, v0, itr)
foldl(op, itr) = mapfoldl(identity, op, itr)
foldr(op, v0, itr) = mapfoldr(identity, op, v0, itr)
foldr(op, itr) = mapfoldr(identity, op, itr)
reduce(op, v0, itr) = mapreduce(identity, op, v0, itr)
reduce(op, itr) = mapreduce(identity, op, itr)
reduce(op, a::Number) = a
prevent_base_import.((:foldl, :foldr, :reduce))

# Implementation of `sum` and `prod` to avoid going into Base. Copied directly from `Base`.
sum(f::Base.Callable, a) = mapreduce(f, +, a)
sum(a) = mapreduce(identity, +, a)
sum(a::AbstractArray{Bool}) = Base.countnz(a)
prod(f::Base.Callable, a) = mapreduce(f, *, a)
prod(a) = mapreduce(identity, *, a)
prevent_base_import.((:sum, :prod))

# # Intercepts for mapreduce under multiplication.
# accepted_mul = :(Tuple{Function, typeof(*), AbstractArray{T} where T<:Real})
# eval(DiffBase, add_intercept(:mapreduce, :(Base.mapreduce), accepted_mul))
# function ∇(
#     ::typeof(mapreduce),
#     ::Type{Arg{3}},
#     p, y, ȳ, f,
#     ::typeof(*),
#     A::AbstractArray{T} where T<:Real,
# )
#     if needs_output(f) && method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, Real, Any})
#         return Base.broadcast(An->ȳ * ∇(f, Arg{1}, An, f(An)), A)
#     elseif !needs_output(f) && method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, Real})
#         return Base.broadcast(An->ȳ * ∇(f, Arg{1}, An), A)
#     else
#         return Base.broadcast(An->ȳ * )
#         throw(error("Not implemented mapreduce sensitivities for general f. ($f)"))
#     end
# end
