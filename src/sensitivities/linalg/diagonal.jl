import Base: det, logdet, diagm, Diagonal, diag
export diag, diagm, Diagonal

const ∇ScalarDiag = Diagonal{<:∇Scalar}

@explicit_intercepts diag Tuple{∇AbstractMatrix}
function ∇(
    ::typeof(diag),
    ::Type{Arg{1}},
    p,
    y::∇AbstractVector,
    ȳ::∇AbstractVector,
    x::∇AbstractMatrix,
)
    x̄ = zeros(x)
    x̄[diagind(x̄)] = ȳ
    return x̄
end
function ∇(
    x̄::∇AbstractMatrix,
    ::typeof(diag),
    ::Type{Arg{1}},
    p,
    y::∇AbstractVector,
    ȳ::∇AbstractVector,
    x::∇AbstractMatrix,
)
    x̄_diag = view(x̄, diagind(x̄))
    x̄_diag .+= ȳ
    return x̄
end

@explicit_intercepts diag Tuple{∇AbstractMatrix, Integer} [true, false]
function ∇(
    ::typeof(diag),
    ::Type{Arg{1}},
    p,
    y::∇AbstractVector,
    ȳ::∇AbstractVector,
    x::∇AbstractMatrix,
    k::Integer,
)
    x̄ = zeros(x)
    x̄[diagind(x̄, k)] = ȳ
    return x̄
end
function ∇(
    x̄::∇AbstractMatrix,
    ::typeof(diag),
    ::Type{Arg{1}},
    p,
    y::∇AbstractVector,
    ȳ::∇AbstractVector,
    x::∇AbstractMatrix,
    k::Integer,
)
    x̄_diag = view(x̄, diagind(x̄, k))
    x̄_diag .+= ȳ
    return x̄
end

@explicit_intercepts diagm Tuple{∇AbstractVector}
function ∇(
    ::typeof(diagm),
    ::Type{Arg{1}},
    p,
    Y::∇AbstractMatrix,
    Ȳ::∇AbstractMatrix,
    x::∇AbstractVector,
)
    return copy!(similar(x), view(Ȳ, diagind(Ȳ)))
end
function ∇(
    x̄::∇AbstractVector,
    ::typeof(diagm),
    ::Type{Arg{1}},
    p,
    Y::∇AbstractMatrix,
    Ȳ::∇AbstractMatrix,
    x::∇AbstractVector,
)
    return broadcast!(+, x̄, x̄, view(Ȳ, diagind(Ȳ)))
end

@explicit_intercepts diagm Tuple{∇AbstractVector, Integer} [true, false]
function ∇(
    ::typeof(diagm),
    ::Type{Arg{1}},
    p,
    Y::∇AbstractMatrix,
    Ȳ::∇AbstractMatrix,
    x::∇AbstractVector,
    k::Integer,
)
    return copy!(similar(x), view(Ȳ, diagind(Ȳ, k)))
end
function ∇(
    x̄::∇AbstractVector,
    ::typeof(diagm),
    ::Type{Arg{1}},
    p,
    Y::∇AbstractMatrix,
    Ȳ::∇AbstractMatrix,
    x::∇AbstractVector,
    k::Integer,
)
    return broadcast!(+, x̄, x̄, view(Ȳ, diagind(Ȳ, k)))
end

@explicit_intercepts diagm Tuple{∇Scalar}
function ∇(
    ::typeof(diagm),
    ::Type{Arg{1}},
    p,
    Y::∇AbstractMatrix,
    Ȳ::∇AbstractMatrix,
    x::∇Scalar,
)
    length(Ȳ) != 1 && throw(error("Ȳ isn't a 1x1 matrix."))
    return Ȳ[1]
end

@explicit_intercepts Diagonal Tuple{∇AbstractVector}
function ∇(
    ::Type{Diagonal},
    ::Type{Arg{1}},
    p,
    Y::∇ScalarDiag,
    Ȳ::∇ScalarDiag,
    x::∇AbstractVector,
)
    return copy!(similar(x), Ȳ.diag)
end
function ∇(
    x̄::∇AbstractVector,
    ::Type{Diagonal},
    ::Type{Arg{1}},
    p,
    Y::∇ScalarDiag,
    Ȳ::∇ScalarDiag,
    x::∇AbstractVector,
)
    return broadcast!(+, x̄, x̄, Ȳ.diag)
end

@explicit_intercepts Diagonal Tuple{∇AbstractMatrix}
function ∇(
    ::Type{Diagonal},
    ::Type{Arg{1}},
    p,
    Y::∇ScalarDiag,
    Ȳ::∇ScalarDiag,
    X::∇AbstractMatrix,
)
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
    X::∇AbstractMatrix,
)
    X̄_diag = view(X̄, diagind(X̄))
    broadcast!(+, X̄_diag, X̄_diag, Ȳ.diag)
    return X̄
end

@explicit_intercepts det Tuple{Diagonal{<:∇Scalar}}
∇(::typeof(det), ::Type{Arg{1}}, p, y::∇Scalar, ȳ::∇Scalar, X::∇ScalarDiag) =
    Diagonal(ȳ .* y ./ X.diag)
function ∇(
    X̄::∇ScalarDiag,
    ::typeof(det),
    ::Type{Arg{1}},
    p,
    y::∇Scalar,
    ȳ::∇Scalar,
    X::∇ScalarDiag,
)
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
