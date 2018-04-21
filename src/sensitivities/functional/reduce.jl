export mapreduce, mapfoldl, mapfoldr

# Intercepts for `mapreduce`, `mapfoldl` and `mapfoldr` under `op` `+`.
type_tuple = :(Tuple{Any, typeof(+), ∇ArrayOrScalar})
for f in (:mapreduce, :mapfoldl, :mapfoldr)
    @eval import Base.$f
    @eval @primitive $f(x...) where __CONTEXT__ <: ∇Ctx = propagate_forward($f, x...)
    @eval ∇(::typeof($f), ::Type{Val{3}}, p, y, ȳ, f, ::typeof(Base.add_sum), A::∇ArrayOrScalar) =
        hasmethod(∇, Tuple{typeof(f), Type{Val{1}}, Real}) ?
            broadcast(An->ȳ * ∇(f, Val{1}, An), A) :
            broadcast(An->ȳ * fmad(f, (An,), Val{1}), A)
    @eval ∇(::typeof($f), ::Type{Val{3}}, p, y, ȳ, f, ::typeof(+), A::∇ArrayOrScalar) =
        hasmethod(∇, Tuple{typeof(f), Type{Val{1}}, Real}) ?
            broadcast(An->ȳ * ∇(f, Val{1}, An), A) :
            broadcast(An->ȳ * fmad(f, (An,), Val{1}), A)
end

import Base: mapreduce
@∇primitive Base.mapreduce


accept_wo_default = :(Tuple{Function, typeof(+), AbstractArray{<:∇Scalar}, Any})
accept_w_default = :(Tuple{Function, typeof(+), AbstractArray{<:∇Scalar}, Any, ∇Scalar})
@primtive mapreducedim(x...) where __CONTEXT__ <: ∇Ctx = propagate_forward(mapreducedim, x...)

∇(::typeof(mapreducedim),
    ::Type{Val{3}},
    p, y, ȳ, f,
    ::typeof(+),
    A::AbstractArray{<:∇Scalar},
    region,
    v0=nothing,
) = method_exists(∇, Tuple{typeof(f), Type{Val{1}}, ∇Scalar}) ?
        broadcast((An, ȳn)->ȳn * ∇(f, Val{1}, An), A, ȳ) :
        broadcast((An, ȳn)->ȳn * fmad(f, (An,), Val{1}), A, ȳ)

# Make `sum` work. It currently fails as the type specification is too restrictive.
sum(n::Node{<:AbstractArray}, region) = mapreducedim(identity, +, n, region)
