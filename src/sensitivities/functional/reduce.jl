export mapreduce, mapfoldl, mapfoldr, reduce, foldl, foldr, sum, prod, mapreducedim

import Nabla.DiffBase.fmad

# Intercepts for `mapreduce`, `mapfoldl` and `mapfoldr` under `op` `+`.
accepted_add = :(Tuple{Any, typeof(+), AbstractArray{T} where T<:Real})
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
        A::AbstractArray{T} where T<:Real,
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

# Intercepts and sensitivities for mapreducedim.
accepted_wo_default = :(Tuple{Function, typeof(+), AbstractArray{T} where T<:Real, Any})
accepted_w_default = :(Tuple{Function, typeof(+), AbstractArray{T} where T<:Real, Any, Real})
eval(DiffBase, add_intercept(:mapreducedim, :(Base.mapreducedim), accepted_wo_default))
eval(DiffBase, add_intercept(:mapreducedim, :(Base.mapreducedim), accepted_w_default))

# Sensitivity w.r.t. mapreducedim.
function ∇(
    ::typeof(mapreducedim),
    ::Type{Arg{3}},
    p, y, ȳ, f,
    ::typeof(+),
    A::AbstractArray{T} where T<:Real,
    region,
)
    if needs_output(f) && method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, Real, Any})
        return Base.broadcast((An, ȳn)->ȳn * ∇(f, Arg{1}, An, f(An)), A, ȳ)
    elseif !needs_output(f) && method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, Real})
        return Base.broadcast((An, ȳn)->ȳn * ∇(f, Arg{1}, An), A, ȳ)
    else
        return Base.broadcast((An, ȳn)->ȳn * fmad(f, (An,), Val{1}), A, ȳ)
    end
end

# Sensitivity w.r.t. mapreducedim with default argument.
function ∇(
    ::typeof(mapreducedim),
    ::Type{Arg{3}},
    p, y, ȳ, f,
    ::typeof(+),
    A::AbstractArray{T} where T<:Real,
    region,
    v0::Real,
)
    if needs_output(f) && method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, Real, Any})
        return Base.broadcast((An, ȳn)->ȳn * ∇(f, Arg{1}, An, f(An)), A, ȳ)
    elseif !needs_output(f) && method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, Real})
        return Base.broadcast((An, ȳn)->ȳn * ∇(f, Arg{1}, An), A, ȳ)
    else
        return Base.broadcast((An, ȳn)->ȳn * fmad(f, (An,), Val{1}), A, ȳn)
    end
end
