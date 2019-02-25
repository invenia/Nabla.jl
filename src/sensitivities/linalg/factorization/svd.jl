import LinearAlgebra: svd
import Base: getproperty

@explicit_intercepts svd Tuple{AbstractMatrix{<:Real}}

∇(::typeof(svd), ::Type{Arg{1}}, p, USV::SVD, S̄::AbstractVector, A::AbstractMatrix) =
    svd_rev(USV, zeroslike(USV.U), S̄, zeroslike(USV.V))::AbstractMatrix
∇(::typeof(svd), ::Type{Arg{1}}, p, USV::SVD, V̄::Adjoint, A::AbstractMatrix) =
    svd_rev(USV, zeroslike(USV.U), zeroslike(USV.S), V̄)::AbstractMatrix
∇(::typeof(svd), ::Type{Arg{1}}, p, USV::SVD, Ū::AbstractMatrix, A::AbstractMatrix) =
    svd_rev(USV, Ū, zeroslike(USV.S), zeroslike(USV.V))::AbstractMatrix

@explicit_intercepts getproperty Tuple{SVD, Symbol} [true, false]

function ∇(::typeof(getproperty), ::Type{Arg{1}}, p, y, ȳ, USV::SVD, x::Symbol)::AbstractMatrix
    if x === :S
        return vec(ȳ)
    elseif x === :U
        return reshape(ȳ, size(USV.U))
    elseif x === :V
        # This is so we can ensure that the result is an Adjoint, otherwise dispatch
        # won't work properly
        return copy(ȳ')'
    elseif x === :Vt
        throw(ArgumentError("Vt is unsupported; use V and transpose the result"))
    else
        throw(ArgumentError("unrecognized property $x; expected U, S, or V"))
    end
end

"""
    svd_rev(USV, Ū, S̄, V̄)

Compute the reverse mode sensitivities of the singular value decomposition (SVD). `USV` is
an `SVD` factorization object produced by a call to `svd`, and `Ū`, `S̄`, and `V̄` are the
respective sensitivities of the `U`, `S`, and `V` factors.
"""
function svd_rev(USV::SVD, Ū::AbstractMatrix, s̄::AbstractVector, V̄::AbstractMatrix)::AbstractMatrix
    # Note: assuming a thin factorization, i.e. svd(A, full=false), which is the default
    U = USV.U
    s = USV.S
    V = USV.V
    Vt = USV.Vt

    k = length(s)
    T = eltype(s)
    F = T[i == j ? 1 : inv(@inbounds s[j]^2 - s[i]^2) for i = 1:k, j = 1:k]

    # We do a lot of matrix operations here, so we'll try to be memory-friendly and do
    # as many of the computations in-place as possible. Benchmarking shows that the in-
    # place functions here are significantly faster than their out-of-place, naively
    # implemented counterparts, and allocate no additional memory.
    Ut = U'
    FUᵀŪ = mulsubtrans!(Ut*Ū, F)  # F .* (UᵀŪ - ŪᵀU)
    FVᵀV̄ = mulsubtrans!(Vt*V̄, F)  # F .* (VᵀV̄ - V̄ᵀV)
    ImUUᵀ = eyesubx!(U*Ut)        # I - UUᵀ
    ImVVᵀ = eyesubx!(V*Vt)        # I - VVᵀ

    S = Diagonal(s)
    S̄ = Diagonal(s̄)

    Ā = add!(U*FUᵀŪ*S, ImUUᵀ*(Ū/S))*Vt
    add!(Ā, U*S̄*Vt)
    add!(Ā, U*add!(S*FVᵀV̄*Vt, (S\V̄')*ImVVᵀ))

    return Ā
end

"""
    mulsubtrans!(X::AbstractMatrix, F::AbstractMatrix)

Compute `F .* (X - X')`, overwriting `X` in the process.

!!! note
    This is an internal function that does no argument checking; the matrices passed to
    this function are square with matching dimensions by construction.
"""
function mulsubtrans!(X::AbstractMatrix{T}, F::AbstractMatrix{T}) where T<:Real
    k = size(X, 1)
    @inbounds for j = 1:k, i = 1:j  # Iterate the upper triangle
        if i == j
            X[i,i] = zero(T)
        else
            X[i,j], X[j,i] = F[i,j] * (X[i,j] - X[j,i]), F[j,i] * (X[j,i] - X[i,j])
        end
    end
    X
end

"""
    eyesubx!(X::AbstractMatrix)

Compute `I - X`, overwriting `X` in the process.
"""
function eyesubx!(X::AbstractMatrix{T}) where T<:Real
    n, m = size(X)
    @inbounds for j = 1:m, i = 1:n
        X[i,j] = (i == j) - X[i,j]
    end
    X
end

"""
    add!(X::AbstractMatrix, Y::AbstractMatrix)

Compute `X + Y`, overwriting X in the process.

!!! note
    This is an internal function that does no argument checking; the matrices passed to
    this function are square with matching dimensions by construction.
"""
function add!(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where T<:Real
    @inbounds for i = eachindex(X, Y)
        X[i] += Y[i]
    end
    X
end
