import Base: size, length, reshape

# Let the user get the `size` and `length` of `Node`s.
Base.size(x::Node, dims...) = size(x.val, dims...)
Base.length(x::Node) = length(x.val)

# Sensitivity for the first argument of `reshape`.
@explicit_intercepts reshape Tuple{∇Array, Vararg{Int}} [true, false]
@explicit_intercepts reshape Tuple{∇Array, Tuple{Vararg{Int}}} [true, false]
∇(::typeof(reshape), ::Type{Arg{1}}, _, y, ȳ, A::∇Array, args...) =
    reshape(ȳ, size(A)...)
