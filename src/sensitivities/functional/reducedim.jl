import Base: mapreducedim, sum

accept_wo_default = :(Tuple{Function, typeof(+), AbstractArray{<:∇Scalar}, Any})
accept_w_default = :(Tuple{Function, typeof(+), AbstractArray{<:∇Scalar}, Any, ∇Scalar})
@eval @explicit_intercepts mapreducedim $accept_wo_default [false, false, true, false]
@eval @explicit_intercepts mapreducedim $accept_w_default [false, false, true, false, true]

∇(::typeof(mapreducedim),
    ::Type{Arg{3}},
    p, y, ȳ, f,
    ::typeof(+),
    A::AbstractArray{<:∇Scalar},
    region,
    v0=nothing,
) = method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, ∇Scalar}) ?
        broadcast((An, ȳn)->ȳn * ∇(f, Arg{1}, An), A, ȳ) :
        broadcast((An, ȳn)->ȳn * fmad(f, (An,), Val{1}), A, ȳ)

# Make `sum` work. It currently fails as the type specification is too restrictive.
sum(n::Node{<:AbstractArray}, region) = mapreducedim(identity, +, n, region)
