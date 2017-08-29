import Base: det, logdet, diagm, Diagonal
export diagm, Diagonal

const ∇RealDiag = Diagonal{<:∇Real}

@explicit_intercepts diagm Tuple{∇RealAV}
∇(::typeof(diagm), ::Type{Arg{1}}, p, Y::∇RealAM, Ȳ::∇RealAM, x::∇RealAV) =
    copy!(similar(x), view(Ȳ, diagind(Ȳ)))
∇(x̄::∇RealAV, ::typeof(diagm), ::Type{Arg{1}}, p, Y::∇RealAM, Ȳ::∇RealAM, x::∇RealAV) =
    broadcast!(+, x̄, x̄, view(Ȳ, diagind(Ȳ)))

@explicit_intercepts Diagonal Tuple{∇RealAV}
∇(::Type{Diagonal}, ::Type{Arg{1}}, p, Y::∇RealDiag, Ȳ::∇RealDiag, x::∇RealAV) =
    copy!(similar(x), Ȳ.diag)
∇(x̄::∇RealAV, ::Type{Diagonal}, ::Type{Arg{1}}, p, Y::∇RealDiag, Ȳ::∇RealDiag, x::∇RealAV) =
    broadcast!(+, x̄, x̄, Ȳ.diag)

@explicit_intercepts Diagonal Tuple{∇RealAM}
function ∇(::Type{Diagonal}, ::Type{Arg{1}}, p, Y::∇RealDiag, Ȳ::∇RealDiag, X::∇RealAM)
    X̄ = zeros(X)
    copy!(view(X̄, diagind(X)), Ȳ.diag)
    return X̄
end
function ∇(
    X̄::∇RealAM,
    ::Type{Diagonal},
    ::Type{Arg{1}},
    p,
    Y::∇RealDiag,
    Ȳ::∇RealDiag,
    X::∇RealAM
)
    X̄_diag = view(X̄, diagind(X̄))
    broadcast!(+, X̄_diag, X̄_diag, Ȳ.diag)
    return X̄
end

@explicit_intercepts det Tuple{Diagonal{<:∇Real}}
∇(::typeof(det), ::Type{Arg{1}}, p, y::∇Real, ȳ::∇Real, X::∇RealDiag) =
    Diagonal(ȳ .* y ./ X.diag)
function ∇(X̄::∇RealDiag, ::typeof(det), ::Type{Arg{1}}, p, y::∇Real, ȳ::∇Real, X::∇RealDiag)
    broadcast!((x̄, x, y, ȳ)->x̄ + ȳ * y / x, X̄.diag, X̄.diag, X.diag, y, ȳ)
    return X̄
end

@explicit_intercepts logdet Tuple{Diagonal{<:∇Real}}
∇(::typeof(logdet), ::Type{Arg{1}}, p, y::∇Real, ȳ::∇Real, X::∇RealDiag) =
    Diagonal(y ./ X.diag)
function ∇(
    X̄::∇RealDiag,
    ::typeof(logdet),
    ::Type{Arg{1}},
    p,
    y::∇Real,
    ȳ::∇Real,
    X::∇RealDiag,
)
    broadcast!((x̄, x, ȳ)->x̄ + ȳ / x, X̄.diag, X̄.diag, X.diag, ȳ)
    return X̄
end
