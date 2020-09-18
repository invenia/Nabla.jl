# Implementation of sensitivities for unary linalg optimisations.
_ϵ, lb, ub = 3e-2, -3.0, 3.0
unary_linalg_optimisations = [#==
    (:-,          ∇Array,  ∇Array,  :(-Ȳ),                               (lb, ub)),
    (:tr,         ∇Array,  ∇Scalar, :(Diagonal(fill!(similar(X), Ȳ))),   (lb, ub)),
    (:inv,        ∇Array,  ∇Array,  :(-transpose(Y) * Ȳ * transpose(Y)), (lb, ub)),
    (:det,        ∇Array,  ∇Scalar, :(Y * Ȳ * transpose(inv(X))),        (_ϵ, ub)),
    (:logdet,     ∇Array,  ∇Scalar, :(Ȳ * transpose(inv(X))),            (_ϵ, ub)),
    (:transpose,  ∇Array,  ∇Array,  :(transpose(Ȳ)),                     (lb, ub)),
    (:adjoint,    ∇Scalar, ∇Scalar, :(adjoint(Ȳ)),                       (_ϵ, ub)),
    (:adjoint,    ∇Array,  ∇Array,  :(adjoint(Ȳ)),                       (lb, ub)),
    (:norm,       ∇Array,  ∇Scalar, :(Ȳ ./ Y .* abs2.(X) ./ X),          (lb, ub)),
    (:norm,       ∇Scalar, ∇Scalar, :(Ȳ * sign(X)),                      (lb, ub))
    ==#
]
for (f, T_In, T_Out, X̄, bounds) in unary_linalg_optimisations
    if f === :-
        @eval import Base: -
    else
        @eval import LinearAlgebra: $f
    end
    @eval begin
        @explicit_intercepts $f Tuple{$T_In}
        ∇(::typeof($f), ::Type{Arg{1}}, p, Y::$T_Out, Ȳ::$T_Out, X::$T_In) = $X̄
    end
end

# Implementation of sensitivities for binary linalg optimisations.
const A = ∇Array
const S = ∇Scalar
const AS = Union{∇Scalar, ∇Array}
δ = 1e-5
binary_linalg_optimisations = [
    (:*, A, A, AS,
        :(Ȳ * B'),
        :(A' * Ȳ)),
    (:/, A, A, AS,
        :(Ȳ / transpose(B)),
        :(-transpose(Y) * (Ȳ / transpose(B)))),
    (:\, A, A, AS,
        :(-(transpose(A) \ Ȳ) * transpose(Y)),
        :(transpose(A) \ Ȳ)),
    (:norm, A, S, S,
        :(Ȳ .* Y^(1 - B) .* abs.(A).^B ./ A),
        :(Ȳ * (Y^(1 - B) * sum(abs.(A).^B .* log.(abs.(A))) - Y * log(Y)) / B)),
    (:norm, S, S, S,
        :(Ȳ * sign(A)),
        :(0)),
]
import Base: *, /, \
import LinearAlgebra: norm
for (f, T_A, T_B, T_Y, Ā, B̄) in binary_linalg_optimisations
    @eval begin
        @explicit_intercepts $f Tuple{$T_A, $T_B}
        ∇(::typeof($f), ::Type{Arg{1}}, p, Y::$T_Y, Ȳ::$T_Y, A::$T_A, B::$T_B) = $Ā
        ∇(::typeof($f), ::Type{Arg{2}}, p, Y::$T_Y, Ȳ::$T_Y, A::$T_A, B::$T_B) = $B̄
    end
end

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

# Short-form `dot`.
@explicit_intercepts LinearAlgebra.dot Tuple{∇Array, ∇Array}
∇(::typeof(LinearAlgebra.dot), ::Type{Arg{1}}, p, z, z̄, x::A, y::A) = z̄ .* y
∇(::typeof(LinearAlgebra.dot), ::Type{Arg{2}}, p, z, z̄, x::A, y::A) = z̄ .* x
∇(x̄, ::typeof(LinearAlgebra.dot), ::Type{Arg{1}}, p, z, z̄, x::A, y::A) = (x̄ .= x̄ .+ z̄ .* y)
∇(ȳ, ::typeof(LinearAlgebra.dot), ::Type{Arg{2}}, p, z, z̄, x::A, y::A) = (ȳ .= ȳ .+ z̄ .* x)

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
