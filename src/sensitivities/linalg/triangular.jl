import LinearAlgebra: det, logdet, LowerTriangular, UpperTriangular

const ∇ScalarLT = LowerTriangular{<:∇Scalar}
const ∇ScalarUT = UpperTriangular{<:∇Scalar}

for (ctor, T) in zip([:LowerTriangular, :UpperTriangular], [:∇ScalarLT, :∇ScalarUT])

    @eval @generated function is_atom(ctx::∇Ctx, ::typeof($ctor), X::∇MaybeTaggedMat)
        return istaggedtype(X, ctx)
    end
    @eval ∇(::Type{$ctor}, ::Type{Arg{1}}, p, Y::$T, Ȳ::$T, X::∇AbstractMatrix) = Matrix(Ȳ)
    @eval ∇(
        X̄::∇AbstractMatrix,
        ::Type{$ctor},
        ::Type{Arg{1}},
        p,
        Y::$T,
        Ȳ::$T,
        X::∇AbstractMatrix,
    ) = broadcast!(+, X̄, X̄, Ȳ)

    @eval @generated function is_atom(ctx::∇Ctx, ::typeof(det), X::∇MaybeTagged{<:$T})
        return istaggedtype(X, ctx)
    end
    @eval ∇(::typeof(det), ::Type{Arg{1}}, p, y::∇Scalar, ȳ::∇Scalar, X::$T) =
        Diagonal(ȳ .* y ./ view(X, diagind(X)))

    # Optimisation for in-place updates.
    @eval function ∇(
        X̄::$T,
        ::typeof(det),
        ::Type{Arg{1}},
        p,
        y::∇Scalar,
        ȳ::∇Scalar,
        X::$T,
    )
        X̄_diag = view(X̄, diagind(X̄))
        broadcast!((x̄, x, y, ȳ)->x̄ + ȳ * y / x,
                   X̄_diag, X̄_diag, view(X, diagind(X)), y, ȳ)
        return X̄
    end

    # Optimisation for in-place updates to `Diagonal` sensitivity cache.
    @eval function ∇(
        X̄::Diagonal,
        ::typeof(det),
        ::Type{Arg{1}},
        p,
        y::∇Scalar,
        ȳ::∇Scalar,
        X::$T,
    )
        X̄.diag .+= ȳ .* y ./ view(X, diagind(X))
        return X̄
    end

    @eval @generated function is_atom(ctx::∇Ctx, ::typeof(logdet), X::∇MaybeTagged{<:$T})
        return istaggedtype(X, ctx)
    end
    @eval ∇(::typeof(logdet), ::Type{Arg{1}}, p, y::∇Scalar, ȳ::∇Scalar, X::$T) =
        Diagonal(ȳ ./ view(X, diagind(X)))

    # Optimisation for in-place updates.
    @eval function ∇(
        X̄::∇Array,
        ::typeof(logdet),
        ::Type{Arg{1}},
        p,
        y::∇Scalar,
        ȳ::∇Scalar,
        X::$T
    )
        X̄_diag = view(X̄, diagind(X̄))
        broadcast!((x̄, x, ȳ)->x̄ + ȳ / x, X̄_diag, X̄_diag, view(X, diagind(X)), ȳ)
        return X̄
    end

    # Optimisation for in-place updates to `Diagonal` sensitivity cache.
    @eval function ∇(
        X̄::Diagonal,
        ::typeof(logdet),
        ::Type{Arg{1}},
        p,
        y::∇Scalar,
        ȳ::∇Scalar,
        X::$T,
    )
        X̄.diag .+= ȳ ./ view(X, diagind(X))
        return X̄
    end
end
