import Base: mapreduce, sum

@explicit_intercepts(
    mapreduce,
    Tuple{Function, Union{typeof(+), typeof(Base.add_sum)}, AbstractArray{<:∇Scalar}},
    [false, false, true],
    (dims=:, init=nothing),
)
function ∇(
    ::typeof(mapreduce),
    ::Type{Arg{3}},
    p, y, ȳ, f,
    ::Union{typeof(+), typeof(Base.add_sum)},
    A::AbstractArray{<:∇Scalar};
    dims=:,
    init=nothing,
)
    hasmethod(∇, Tuple{typeof(f), Type{Arg{1}}, ∇Scalar}) ?
        broadcast((An, ȳn)->ȳn * ∇(f, Arg{1}, An), A, ȳ) :
        broadcast((An, ȳn)->ȳn * fmad(f, (An,), Val{1}), A, ȳ)
end

# Make `sum` work. It currently fails as the type specification is too restrictive.
sum(n::Node{<:AbstractArray}; dims=:) = mapreduce(identity, Base.add_sum, n, dims=dims)
