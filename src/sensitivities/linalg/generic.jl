# Implementation of sensitivities for unary linalg optimisations.
# Implementation of sensitivities for binary linalg optimisations.
const A = ∇Array
const S = ∇Scalar
const AS = Union{∇Scalar, ∇Array}


# Sensitivities for the Kronecker product:
import LinearAlgebra: kron
@explicit_intercepts kron Tuple{A, A}

# The allocating versions simply allocate and then call the in-place versions.
∇(::typeof(kron), ::Type{Arg{1}}, p, Y::A, Ȳ::A, A::A, B::A) =
    ∇(zeroslike(A), kron, Arg{1}, p, Y, Ȳ, A, B)
∇(::typeof(kron), ::Type{Arg{2}}, p, Y::A, Ȳ::A, A::A, B::A) =
    ∇(zeroslike(B), kron, Arg{2}, p, Y, Ȳ, A, B)

function ∇(Ā::A, ::typeof(kron), ::Type{Arg{1}}, p, Y::A, Ȳ::A, A::A, B::A)
    (I, J), (K, L), m = size(A), size(B), length(Y)
    for j = reverse(1:J), l = reverse(1:L), i = reverse(1:I)
        aij, āij = A[i, j], Ā[i, j]
        for k = reverse(1:K)
            āij += Ȳ[m] * B[k, l]
            m -= 1
        end
        Ā[i, j] = āij
    end
    return Ā
end
function ∇(B̄::A, ::typeof(kron), ::Type{Arg{2}}, p, Y::A, Ȳ::A, A::A, B::A)
    (I, J), (K, L), m = size(A), size(B), length(Y)
    for j = reverse(1:J), l = reverse(1:L), i = reverse(1:I)
        aij = A[i, j]
        for k = reverse(1:K)
            B̄[k, l] += Ȳ[m] * aij
            m -= 1
        end
    end
    return B̄
end

@explicit_intercepts Base.:+ Tuple{A, UniformScaling}
∇(::typeof(+), ::Type{Arg{1}}, p, Y::∇Array, Ȳ::∇Array, A::∇Array, B::UniformScaling) = Ȳ

@explicit_intercepts Base.:+ Tuple{UniformScaling, A}
∇(::typeof(+), ::Type{Arg{2}}, p, Y::∇Array, Ȳ::∇Array, A::UniformScaling, B::∇Array) = Ȳ

# `copy` materializes `Adjoint` and `Transpose` wrappers but can be called on anything
import Base: copy
@explicit_intercepts copy Tuple{Any}
∇(::typeof(copy), ::Type{Arg{1}}, p, Y, Ȳ, A) = copy(Ȳ)

# Matrix exponential
# Ported from Theano, see https://github.com/Theano/Theano/blob/3b8a5b342b30c7ffd2f89f0...
# e9efef601b7492411/theano/tensor/slinalg.py#L518-L553
# Implementation there is based on Kalbfleisch and Lawless, 1985, The Analysis of Panel
# Data Under a Markov Assumption.
import Base: exp
@explicit_intercepts exp Tuple{AbstractMatrix{<:∇Scalar}}
function ∇(::typeof(exp), ::Type{Arg{1}}, p, Y, Ȳ, X::AbstractMatrix)
    # TODO: Make this work for asymmetric matrices
    issymmetric(X) || throw(ArgumentError("input is not symmetric; eigenvalues are complex"))
    n = LinearAlgebra.checksquare(X)
    λ, U = eigen(X)
    eλ = exp.(λ)
    Z = @inbounds begin
        eltype(eλ)[i == j ? eλ[i] : (eλ[i] - eλ[j]) / (λ[i] - λ[j]) for i = 1:n, j = 1:n]
    end
    Uᵀ = transpose(U)
    F = factorize(Uᵀ)
    return real(F \ (Uᵀ * Ȳ / F .* Z) * Uᵀ)
end
