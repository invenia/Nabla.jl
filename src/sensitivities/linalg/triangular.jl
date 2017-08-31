import Base: det, logdet, LowerTriangular, UpperTriangular
export det, logdet, LowerTriangular, UpperTriangular

const ∇RealLT = LowerTriangular{<:∇Real}
const ∇RealUT = UpperTriangular{<:∇Real}

for (ctor, T) in zip([:LowerTriangular, :UpperTriangular], [:∇RealLT, :∇RealUT])

    @eval @explicit_intercepts $ctor Tuple{∇RealAM}
    @eval ∇(::Type{$ctor}, ::Type{Arg{1}}, p, Y::$T, Ȳ::$T, X::∇RealAM) = full(Ȳ)
    @eval ∇(X̄::∇RealAM, ::Type{$ctor}, ::Type{Arg{1}}, p, Y::$T, Ȳ::$T, X::∇RealAM) =
        broadcast!(+, X̄, X̄, Ȳ)

    @eval @explicit_intercepts det Tuple{$T}
    @eval function ∇(::typeof(det), ::Type{Arg{1}}, p, y::∇Real, ȳ::∇Real, X::$T)
        data = zeros(X.data)
        data[diagind(data)] .= ȳ .* y ./ view(X, diagind(X))
        return $ctor(data)
    end
    @eval function ∇(X̄::$T, ::typeof(det), ::Type{Arg{1}}, p, y::∇Real, ȳ::∇Real, X::$T)
        X̄_diag = view(X̄, diagind(X̄))
        broadcast!((x̄, x, y, ȳ)->x̄ + ȳ * y / x, X̄_diag, X̄_diag, view(X, diagind(X)), y, ȳ)
        return X̄
    end

    @eval @explicit_intercepts logdet Tuple{$T}
    @eval function ∇(::typeof(logdet), ::Type{Arg{1}}, p, y::∇Real, ȳ::∇Real, X::$T)
        data = zeros(X.data)
        data[diagind(data)] .= ȳ ./ view(X, diagind(X))
        return $ctor(data)
    end
    @eval function ∇(X̄::$T, ::typeof(logdet), ::Type{Arg{1}}, p, y::∇Real, ȳ::∇Real, X::$T)
        X̄_diag = view(X̄, diagind(X̄))
        broadcast!((x̄, x, ȳ)->x̄ + ȳ / x, X̄_diag, X̄_diag, view(X, diagind(X)), ȳ)
        return X̄
    end
end
