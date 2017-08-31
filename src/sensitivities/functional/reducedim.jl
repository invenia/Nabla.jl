import Base: mapreducedim, sum

accepted_wo_default = :(Tuple{Function, typeof(+), AbstractArray{<:∇Scalar}, Any})
accepted_w_default = :(Tuple{Function, typeof(+), AbstractArray{<:∇Scalar}, Any, ∇Scalar})
@eval @explicit_intercepts mapreducedim $accepted_wo_default [false, false, true, false]
@eval @explicit_intercepts mapreducedim $accepted_w_default [false, false, true, false, true]

function ∇(
    ::typeof(mapreducedim),
    ::Type{Arg{3}},
    p, y, ȳ, f,
    ::typeof(+),
    A::AbstractArray{<:∇Scalar},
    region,
    v0=nothing,
)
    if needs_output(f) && method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, ∇Scalar, Any})
        return broadcast((An, ȳn)->ȳn * ∇(f, Arg{1}, An, f(An)), A, ȳ)
    elseif !needs_output(f) && method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, ∇Scalar})
        return broadcast((An, ȳn)->ȳn * ∇(f, Arg{1}, An), A, ȳ)
    else
        return broadcast((An, ȳn)->ȳn * fmad(f, (An,), Val{1}), A, ȳ)
    end
end

# Make `sum` work. It currently fails as the type specification is too restrictive.
sum(n::Node{<:AbstractArray}, region) = mapreducedim(identity, +, n, region)
