import Base.mapreducedim
accepted_wo_default = :(Tuple{Function, typeof(+), AbstractArray{<:∇Real}, Any})
accepted_w_default = :(Tuple{Function, typeof(+), AbstractArray{<:∇Real}, Any, ∇Real})
@eval @explicit_intercepts mapreducedim $accepted_wo_default [false, false, true, false]
@eval @explicit_intercepts mapreducedim $accepted_w_default [false, false, true, false, true]

function ∇(
    ::typeof(mapreducedim),
    ::Type{Arg{3}},
    p, y, ȳ, f,
    ::typeof(+),
    A::AbstractArray{<:∇Real},
    region,
    v0=nothing,
)
    if needs_output(f) && method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, ∇Real, Any})
        return Base.broadcast((An, ȳn)->ȳn * ∇(f, Arg{1}, An, f(An)), A, ȳ)
    elseif !needs_output(f) && method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, ∇Real})
        return Base.broadcast((An, ȳn)->ȳn * ∇(f, Arg{1}, An), A, ȳ)
    else
        return Base.broadcast((An, ȳn)->ȳn * fmad(f, (An,), Val{1}), A, ȳ)
    end
end

# Make `sum` work. It currently fails as the type specification is too restrictive.
Base.sum(n::Node, region) = mapreducedim(Base.identity, +, n, region)
