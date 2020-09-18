import Base: mapreduce, sum
import Statistics: mean

@eval begin
    @explicit_intercepts(
        mapreduce,
        Tuple{Function, $plustype, AbstractArray{<:∇Scalar}},
        [false, false, true],
        (dims=:,),
    )
    function ∇(
        ::typeof(mapreduce),
        ::Type{Arg{3}},
        p, y, ȳ, f,
        ::$plustype,
        A::AbstractArray{<:∇Scalar};
        dims=:,
    )
        return broadcast((An, ȳn)->ȳn * ForwardDiff.derivative(f, An), A, ȳ)
    end
end

@explicit_intercepts(
    sum,
    Tuple{Function, AbstractArray{<:∇Scalar}},
    [false, true],
    (dims=:,),
)
function ∇(
    ::typeof(sum),
    ::Type{Arg{2}},
    p, y, ȳ,
    f::Function,
    A::AbstractArray{<:∇Scalar};
    dims=:,
)
    # Just pass through to mapreduce
    return ∇(mapreduce, Arg{3}, p, y, ȳ, f, Base.add_sum, A; dims=dims)
end
# sum(abs2, xs) is in ChainRules, but it results in method ambiguties with the
# version that accepts any function above
function ∇(
    ::typeof(sum),
    ::Type{Arg{2}},
    p, y, ȳ,
    ::typeof(abs2),
    A::AbstractArray{<:Real};
    dims=:,
)
    return 2ȳ .* A
end

@explicit_intercepts(
    mean,
    Tuple{Function, AbstractArray{<:∇Scalar}},
    [false, true],
    #(dims=:,)  # https://github.com/JuliaLang/julia/issues/31412
)

_denom(x, dims::Colon) = length(x)
_denom(x, dims::Integer) = size(x, dims)
_denom(x, dims) = mapreduce(i->size(x, i), Base.mul_prod, unique(dims), init=1)

function ∇(
    ::typeof(mean),
    ::Type{Arg{2}},
    p, y, ȳ,
    f::Function,
    x::AbstractArray{<:∇Scalar},
)
    return ∇(sum, Arg{2}, p, y, ȳ, f, x; dims=:) / _denom(x, :)
end
