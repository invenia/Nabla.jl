import Base: size, length, reshape, hcat, vcat, fill

# Let the user get the `size` and `length` of `Node`s.
Base.size(x::Node, dims...) = size(unbox(x), dims...)
Base.length(x::Node) = length(unbox(x))

# Sensitivity for the first argument of `reshape`.
@explicit_intercepts reshape Tuple{∇Array, Vararg{Int}} [true, false]
@explicit_intercepts reshape Tuple{∇Array, Tuple{Vararg{Int}}} [true, false]
∇(::typeof(reshape), ::Type{Arg{1}}, _, y, ȳ, A::∇Array, args...) =
    reshape(ȳ, size(A)...)

@union_intercepts hcat Tuple{Vararg{∇Array}} Tuple{Vararg{AbstractArray}}
function Nabla.∇(
    ::typeof(hcat),
    ::Type{Arg{i}},
    _,
    y,
    ȳ,
    A::AbstractArray...
) where i
    l = sum([size(A[j], 2) for j in 1:(i - 1)])
    u = l + size(A[i], 2)
    # Using copy materializes the views returned by selectdim
    return copy(u > l + 1 ? selectdim(ȳ, 2, (l+1):u) : selectdim(ȳ, 2, u))
end

@union_intercepts vcat Tuple{Vararg{∇Array}} Tuple{Vararg{AbstractArray}}
function Nabla.∇(
    ::typeof(vcat),
    ::Type{Arg{i}},
    _,
    y,
    ȳ,
    A::AbstractArray...
) where i
    l = sum([size(A[j], 1) for j in 1:(i - 1)])
    u = l + size(A[i], 1)
    return copy(selectdim(ȳ, 1, (l+1):u))
end

@explicit_intercepts fill Tuple{Any, Tuple{Vararg{Integer}}} [true, false]
∇(::typeof(fill), ::Type{Arg{1}}, p, y, ȳ, value, dims...) = sum(ȳ)
