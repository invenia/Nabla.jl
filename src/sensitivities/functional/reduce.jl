# Intercepts for `mapreduce`, `mapfoldl` and `mapfoldr` under `op` `+`.
const plustype = Union{typeof(+), typeof(Base.add_sum)}
const type_tuple = :(Tuple{Any, $plustype, ∇ArrayOrScalar})
for f in (:mapfoldl, :mapfoldr)
    @eval begin
        import Base: $f
        @explicit_intercepts $f $type_tuple [false, false, true] #(init=0,)
        function ∇(::typeof($f), ::Type{Arg{3}}, p, y, ȳ, f, ::$plustype, A::∇ArrayOrScalar)
            return ȳ .* ForwardDiff.derivative.(f, A)
        end
    end
end
