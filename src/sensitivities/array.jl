import Base: size, length, reshape

# Let the user get the `size` and `length` of `Node`s.
Base.size(x::Node, dims...) = size(x.val, dims...)
Base.length(x::Node) = length(x.val)

# Sensitivity for the first argument of `reshape`.
@primitive reshape(args...) where {__CONTEXT__ <: ∇Ctx} = propagate_forward(reshape, args...)
∇(::typeof(reshape), ::Type{Val{1}}, _, y, ȳ, A::∇Array, args...) = reshape(ȳ, size(A)...)
