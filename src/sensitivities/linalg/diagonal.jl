import Base: det, logdet, diagm, Diagonal
export diagm, Diagonal

const ∇ScalarDiag = Diagonal{<:∇Scalar}

@explicit_intercepts diagm Tuple{∇AbstractVector}
∇(::typeof(diagm), ::Type{Arg{1}}, p, Y::∇AbstractMatrix, Ȳ::∇AbstractMatrix, x::∇AbstractVector) =
    copy!(similar(x), view(Ȳ, diagind(Ȳ)))
∇(x̄::∇AbstractVector, ::typeof(diagm), ::Type{Arg{1}}, p, Y::∇AbstractMatrix, Ȳ::∇AbstractMatrix, x::∇AbstractVector) =
    broadcast!(+, x̄, x̄, view(Ȳ, diagind(Ȳ)))

@explicit_intercepts Diagonal Tuple{∇AbstractVector}
∇(::Type{Diagonal}, ::Type{Arg{1}}, p, Y::∇ScalarDiag, Ȳ::∇ScalarDiag, x::∇AbstractVector) =
    copy!(similar(x), Ȳ.diag)
∇(x̄::∇AbstractVector, ::Type{Diagonal}, ::Type{Arg{1}}, p, Y::∇ScalarDiag, Ȳ::∇ScalarDiag, x::∇AbstractVector) =
    broadcast!(+, x̄, x̄, Ȳ.diag)

@explicit_intercepts Diagonal Tuple{∇AbstractMatrix}
function ∇(::Type{Diagonal}, ::Type{Arg{1}}, p, Y::∇ScalarDiag, Ȳ::∇ScalarDiag, X::∇AbstractMatrix)
    X̄ = zeros(X)
    copy!(view(X̄, diagind(X)), Ȳ.diag)
    return X̄
end
function ∇(
    X̄::∇AbstractMatrix,
    ::Type{Diagonal},
    ::Type{Arg{1}},
    p,
    Y::∇ScalarDiag,
    Ȳ::∇ScalarDiag,
    X::∇AbstractMatrix
)
    X̄_diag = view(X̄, diagind(X̄))
    broadcast!(+, X̄_diag, X̄_diag, Ȳ.diag)
    return X̄
end

@explicit_intercepts det Tuple{Diagonal{<:∇Scalar}}
∇(::typeof(det), ::Type{Arg{1}}, p, y::∇Scalar, ȳ::∇Scalar, X::∇ScalarDiag) =
    Diagonal(ȳ .* y ./ X.diag)
function ∇(X̄::∇ScalarDiag, ::typeof(det), ::Type{Arg{1}}, p, y::∇Scalar, ȳ::∇Scalar, X::∇ScalarDiag)
    broadcast!((x̄, x, y, ȳ)->x̄ + ȳ * y / x, X̄.diag, X̄.diag, X.diag, y, ȳ)
    return X̄
end

@explicit_intercepts logdet Tuple{Diagonal{<:∇Scalar}}
∇(::typeof(logdet), ::Type{Arg{1}}, p, y::∇Scalar, ȳ::∇Scalar, X::∇ScalarDiag) =
    Diagonal(ȳ ./ X.diag)
function ∇(
    X̄::∇ScalarDiag,
    ::typeof(logdet),
    ::Type{Arg{1}},
    p,
    y::∇Scalar,
    ȳ::∇Scalar,
    X::∇ScalarDiag,
)
    broadcast!((x̄, x, ȳ)->x̄ + ȳ / x, X̄.diag, X̄.diag, X.diag, ȳ)
    return X̄
end
