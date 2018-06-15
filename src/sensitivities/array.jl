import Base: size, length, reshape, hcat, vcat

# Let the user get the `size` and `length` of `Node`s.
Base.size(x::Node, dims...) = size(x.val, dims...)
Base.length(x::Node) = length(x.val)

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
    return u > l + 1 ? slicedim(ȳ, 2, (l+1):u) : slicedim(ȳ, 2, u)
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
    return slicedim(ȳ, 1, (l+1):u)
end
