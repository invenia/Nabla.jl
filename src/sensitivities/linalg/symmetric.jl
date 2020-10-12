import LinearAlgebra: Symmetric
@explicit_intercepts Symmetric Tuple{∇Array}
∇(::Type{Symmetric}, ::Type{Arg{1}}, p, Y::∇Array, Ȳ::∇Array, X::∇Array) =
    UpperTriangular(Ȳ) + LowerTriangular(Ȳ)' - Diagonal(Ȳ)
