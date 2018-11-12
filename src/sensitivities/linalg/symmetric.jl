import LinearAlgebra: Symmetric
@generated function is_atom(ctx::∇Ctx, ::typeof(Symmetric), X::∇MaybeTagged{<:∇Array})
    return istaggedtype(X, ctx)
end
function ∇(::typeof(Symmetric), ::Type{Arg{1}}, p, Y::∇Array, Ȳ::∇Array, X::∇Array)
    return UpperTriangular(Ȳ) + LowerTriangular(Ȳ)' - Diagonal(Ȳ)
end
