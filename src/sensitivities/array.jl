import Base: size, length, reshape, hcat, vcat

# # Let the user get the `size` and `length` of `Node`s.
# Base.size(x::Node, dims...) = size(x.val, dims...)
# Base.length(x::Node) = length(x.val)

# Sensitivity for the first argument of `reshape`.
# @explicit_intercepts reshape Tuple{∇Array, Vararg{Int}} [true, false]
# @explicit_intercepts reshape Tuple{∇Array, Tuple{Vararg{Int}}} [true, false]
# ∇(::typeof(reshape), ::Type{Arg{1}}, _, y, ȳ, A::∇Array, args...) = reshape(ȳ, size(A)...)

@generated function is_atom(ctx::∇Ctx, ::typeof(hcat), A::∇MaybeTagged{<:AbstractArray}...)
    return any(a->istaggedtype(a, ctx), A)
end
function ∇(::typeof(hcat), ::Type{Arg{i}}, _, y, ȳ, A::AbstractArray...) where i
    l = sum([size(A[j], 2) for j in 1:(i - 1)])
    u = l + size(A[i], 2)

    # Using copy materializes the views returned by selectdim
    return copy(u > l + 1 ? selectdim(ȳ, 2, (l+1):u) : selectdim(ȳ, 2, u))
end

@generated function is_atom(ctx::∇Ctx, ::typeof(vcat), A::∇MaybeTagged{<:AbstractArray}...)
    return any(a->istaggedtype(a, ctx), A)
end
function ∇(::typeof(vcat), ::Type{Arg{i}}, _, y, ȳ, A::AbstractArray...) where i
    l = sum([size(A[j], 1) for j in 1:(i - 1)])
    u = l + size(A[i], 1)
    return copy(selectdim(ȳ, 1, (l+1):u))
end
