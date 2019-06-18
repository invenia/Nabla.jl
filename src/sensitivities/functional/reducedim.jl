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
        hasmethod(∇, Tuple{typeof(f), Type{Arg{1}}, ∇Scalar}) ?
            broadcast((An, ȳn)->ȳn * ∇(f, Arg{1}, An), A, ȳ) :
            broadcast((An, ȳn)->ȳn * fmad(f, (An,), Val{1}), A, ȳ)
    end
end

@explicit_intercepts(
    sum,
    Tuple{Function, AbstractArray{<:∇Scalar}},
    [false, true],
    (dims=:,),
)
@explicit_intercepts(
    sum,
    Tuple{AbstractArray{<:∇Scalar}},
    [true],
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
function ∇(
    ::typeof(sum),
    ::Type{Arg{1}},
    p, y, ȳ,
    A::AbstractArray{<:∇Scalar};
    dims=:,
)
    # Again pass through to mapreduce, using identity as the mapped function
    return ∇(mapreduce, Arg{3}, p, y, ȳ, identity, Base.add_sum, A; dims=dims)
end
# Specialize on sum(abs2, x) as it's a common pattern with a simple derivative
function ∇(
    ::typeof(sum),
    ::Type{Arg{2}},
    p, y, ȳ,
    ::typeof(abs2),
    A::AbstractArray{<:∇Scalar};
    dims=:,
)
    return 2ȳ .* A
end

@explicit_intercepts(
    mean,
    Tuple{Function, AbstractArray{<:∇Scalar}},
    [false, true],
    #(dims=:,)  # https://github.com/JuliaLang/julia/issues/31412
)
@explicit_intercepts(
    mean,
    Tuple{AbstractArray{<:∇Scalar}},
    [true],
    (dims=:,)
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
function ∇(
    ::typeof(mean),
    ::Type{Arg{1}},
    p, y, ȳ,
    x::AbstractArray{<:∇Scalar};
    dims=:,
)
    return ∇(sum, Arg{1}, p, y, ȳ, x; dims=dims) ./ _denom(x, dims)
end
