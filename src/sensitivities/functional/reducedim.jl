export mapreducedim, reducedim, sum, prod

# Intercepts and sensitivities for mapreducedim.
accepted_wo_default = :(Tuple{Function, typeof(+), AbstractArray{<:Real}, Any})
accepted_w_default = :(Tuple{Function, typeof(+), AbstractArray{<:Real}, Any, Real})
eval(DiffBase, add_intercept(:mapreducedim, :(Base.mapreducedim), accepted_wo_default))
eval(DiffBase, add_intercept(:mapreducedim, :(Base.mapreducedim), accepted_w_default))

# Sensitivity w.r.t. mapreducedim.
function ∇(
    ::typeof(mapreducedim),
    ::Type{Arg{3}},
    p, y, ȳ, f,
    ::typeof(+),
    A::AbstractArray{<:Real},
    region,
    v0=nothing,
)
    if needs_output(f) && method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, Real, Any})
        return Base.broadcast((An, ȳn)->ȳn * ∇(f, Arg{1}, An, f(An)), A, ȳ)
    elseif !needs_output(f) && method_exists(∇, Tuple{typeof(f), Type{Arg{1}}, Real})
        return Base.broadcast((An, ȳn)->ȳn * ∇(f, Arg{1}, An), A, ȳ)
    else
        return Base.broadcast((An, ȳn)->ȳn * fmad(f, (An,), Val{1}), A, ȳ)
    end
end

# Implementation of `reducedim`, `sum` and `prod` to avoid going into `Base`.
# Copied directly from `Base`. This is bad and will change.
const A_type = Union{AbstractArray, Node{<:AbstractArray}}
reducedim(op, A::A_type, region, v0) = mapreducedim(identity, op, A, region, v0)
reducedim(op, A::A_type, region) = mapreducedim(identity, op, A, region)
sum(f::Function, A::A_type, region) = mapreducedim(f, +, A, region)
sum(A::A_type, region) = sum(identity, A, region)
prod(f::Function, A::A_type, region) = mapreducedim(f, *, A, region)
prod(A::A_type, region) = prod(identity, A, region)
prevent_base_import.((:reducedim, :sum, :prod))
