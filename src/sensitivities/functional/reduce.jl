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
        if method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, Real})
            return Base.broadcast(An->ȳ * ∇(f, Arg{1}, An), A)
        else
            return Base.broadcast(An->ȳ * fmad(f, (An,), Val{1}), A)
        end
    end
end
