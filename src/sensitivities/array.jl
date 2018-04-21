import Base: size, length, reshape

# Let the user get the `size` and `length` of `Node`s.
Base.size(x::Box{<:∇Ctx}, dims...) = size(x.value, dims...)
Base.length(x::Box{<:∇Ctx}) = length(x.value)

# Sensitivity for the first argument of `reshape`.
@∇primitive Base.reshape
∇(::typeof(reshape), ::Type{Val{1}}, _, y, ȳ, A::∇Array, args...) = reshape(ȳ, size(A)...)
