# Intercepts for `mapreduce`, `mapfoldl` and `mapfoldr` under `op` `+`.
const plustype = Union{typeof(+), typeof(Base.add_sum)}
const type_tuple = :(Tuple{Any, $plustype, ∇ArrayOrScalar})
for f in (:mapfoldl, :mapfoldr)
    @eval begin
        import Base: $f
        @explicit_intercepts $f $type_tuple [false, false, true] #(init=0,)
        ∇(::typeof($f), ::Type{Arg{3}}, p, y, ȳ, f, ::$plustype, A::∇ArrayOrScalar) =
            hasmethod(∇, Tuple{typeof(f), Type{Arg{1}}, Real}) ?
                broadcast(An->ȳ * ∇(f, Arg{1}, An), A) :
                broadcast(An->ȳ * fmad(f, (An,), Val{1}), A)
    end
end
