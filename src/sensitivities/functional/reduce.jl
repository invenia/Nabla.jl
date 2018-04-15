export mapreduce, mapfoldl, mapfoldr

# Intercepts for `mapreduce`, `mapfoldl` and `mapfoldr` under `op` `+`.
type_tuple = :(Tuple{Any, typeof(+), ∇ArrayOrScalar})
for f in (:mapreduce, :mapfoldl, :mapfoldr)
    @eval import Base.$f
    @eval @primitive $f(x...) where __CONTEXT__ <: ∇Ctx = propagate_forward($f, x...)
    @eval ∇(::typeof($f), ::Type{Arg{3}}, p, y, ȳ, f, ::typeof(Base.add_sum), A::∇ArrayOrScalar) =
        hasmethod(∇, Tuple{typeof(f), Type{Arg{1}}, Real}) ?
            broadcast(An->ȳ * ∇(f, Arg{1}, An), A) :
            broadcast(An->ȳ * fmad(f, (An,), Val{1}), A)
    @eval ∇(::typeof($f), ::Type{Arg{3}}, p, y, ȳ, f, ::typeof(+), A::∇ArrayOrScalar) =
        hasmethod(∇, Tuple{typeof(f), Type{Arg{1}}, Real}) ?
            broadcast(An->ȳ * ∇(f, Arg{1}, An), A) :
            broadcast(An->ȳ * fmad(f, (An,), Val{1}), A)
end
