export mapreduce, mapfoldl, mapfoldr, mapreducedim

# Intercepts for `mapreduce`, `mapfoldl` and `mapfoldr` under `op` `+`.
type_tuple = :(Tuple{Any, typeof(+), AbstractArray{<:Real}})
for (f, base_f_name) in ((:mapreduce, :(Base.mapreduce)),
                              (:mapfoldl, :(Base.mapfoldl)),
                              (:mapfoldr, :(Base.mapfoldr)))
    @eval import Base.$f
    @eval @explicit_intercepts $f $type_tuple [false, false, true]
    @eval function ∇(
        ::typeof($f),
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
