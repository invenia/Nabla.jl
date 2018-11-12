# Implementation of sensitivities for unary linalg optimisations.
_ϵ, lb, ub = 3e-2, -3.0, 3.0
unary_linalg_optimisations = [
    (:-,          ∇Array,  ∇Array,  :(map(-, Ȳ)),                        (lb, ub)),
    (:tr,         ∇Array,  ∇Scalar, :(Diagonal(fill!(similar(X), Ȳ))),   (lb, ub)),
    (:inv,        ∇Array,  ∇Array,  :(-transpose(Y) * Ȳ * transpose(Y)), (lb, ub)),
    (:det,        ∇Array,  ∇Scalar, :(Y * Ȳ * transpose(inv(X))),        (_ϵ, ub)),
    (:logdet,     ∇Array,  ∇Scalar, :(Ȳ * transpose(inv(X))),            (_ϵ, ub)),
    (:transpose,  ∇Array,  ∇Array,  :(transpose(Ȳ)),                     (lb, ub)),
    (:adjoint,    ∇Array,  ∇Array,  :(adjoint(Ȳ)),                       (lb, ub)),
    (:norm,       ∇Array,  ∇Scalar, :(Ȳ ./ Y .* abs2.(X) ./ X),          (lb, ub)),
    (:norm,       ∇Scalar, ∇Scalar, :(Ȳ * sign(X)),                      (lb, ub))
]
for (f, T_In, T_Out, X̄, bounds) in unary_linalg_optimisations
    if f === :-
        @eval import Base: -
    else
        @eval import LinearAlgebra: $f
    end
    @eval begin
        @generated function is_atom(ctx::∇Ctx, ::typeof($f), X::∇MaybeTagged{<:$T_In})
            return istaggedtype(X, ctx)
        end
        ∇(::typeof($f), ::Type{Arg{1}}, p, Y::$T_Out, Ȳ::$T_Out, X::$T_In) = $X̄
    end
end

# Implementation of sensitivities for binary linalg optimisations.
const A = ∇Array
const S = ∇Scalar
const AS = Union{∇Scalar, ∇Array}
const AT = Transpose{<:∇Scalar, ∇Array}
const AH = Adjoint{<:∇Scalar, ∇Array}
δ = 1e-5
binary_linalg_optimisations = [
    (:*, A, A, AS,
        :(Ȳ * B'),
        :(A' * Ȳ)),
    (:*, AT, A, AS,
        :(B * transpose(Ȳ)),
        :(A * Ȳ)),
    (:*, A, AT, AS,
        :(Ȳ * B),
        :(transpose(Ȳ) * A)),
    (:*, AT, AT, AS,
        :(transpose(B) * transpose(Ȳ)),
        :(transpose(Ȳ) * transpose(A))),
    (:*, AH, A, AS,
        :(B * transpose(Ȳ)),
        :(A * Ȳ)),
    (:*, A, AH, AS,
        :(Ȳ * B),
        :(Ȳ' * A)),
    (:*, AH, AH, AS,
        :(B' * Ȳ'),
        :(Ȳ' * A')),
    (:/, A, A, AS,
        :(Ȳ / transpose(B)),
        :(-transpose(Y) * (Ȳ / transpose(B)))),
    (:/, AT, A, AS,
        :(B \ transpose(Ȳ)),
        :(-transpose(Y) * (Ȳ / transpose(B)))),
    (:/, A, AT, AS,
        :(Ȳ / B),
        :(-(transpose(B) \ transpose(Ȳ)) * Y)),
    (:/, AT, AT, AS,
        :(transpose(B) \ transpose(Ȳ)),
        :(-(transpose(B) \ transpose(Ȳ)) * Y)),
    (:/, AH, A, AS,
        :(B \ Ȳ'),
        :(-Y' * (Ȳ / B'))),
    (:/, A, AH, AS,
        :(Ȳ / B),
        :(-(transpose(B) \ transpose(Ȳ)) * Y)),
    (:/, AH, AH, AS,
        :(B' \ Ȳ'),
        :(-(B' \ Ȳ') * Y)),
    (:\, A, A, AS,
        :(-(transpose(A) \ Ȳ) * transpose(Y)),
        :(transpose(A) \ Ȳ)),
    (:\, AT, A, AS,
        :(-Y * transpose(A \ Ȳ)),
        :(A \ Ȳ)),
    (:\, A, AT, AS,
        :(-transpose(transpose(Ȳ) / A) * transpose(Y)),
        :(transpose(Ȳ) / A)),
    (:\, AT, AT, AS,
       :(-Y * (transpose(Ȳ) / transpose(A))),
       :(transpose(Ȳ) / transpose(A))),
    (:\, AH, A, AS,
        :(-Y * (A \ Ȳ)'),
        :(A \ Ȳ)),
    (:\, A, AH, AS,
        :(-(Ȳ' / A)' * Y),
        :(Ȳ' / A)),
    (:\, AH, AH, AS,
        :(-Y * (Ȳ' / A')),
        :(Ȳ' / A')),
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
        @generated function is_atom(
            ctx::∇Ctx,
            ::typeof($f),
            A::∇MaybeTagged{<:$T_A},
            B::∇MaybeTagged{<:$T_B},
        )
            return istaggedtype(A, ctx) || istaggedtype(B, ctx)
        end
        ∇(::typeof($f), ::Type{Arg{1}}, p, Y::$T_Y, Ȳ::$T_Y, A::$T_A, B::$T_B) = $Ā
        ∇(::typeof($f), ::Type{Arg{2}}, p, Y::$T_Y, Ȳ::$T_Y, A::$T_A, B::$T_B) = $B̄
    end
end

# Sensitivities for the Kronecker product:
import LinearAlgebra: kron
@generated function is_atom(
    ctx::∇Ctx, ::typeof(kron),
    A::∇MaybeTagged{<:A},
    B::∇MaybeTagged{<:A},
)
    return istaggedtype(A, ctx) || istaggedtype(B, ctx)
end

# The allocating versions simply allocate and then call the in-place versions.
function ∇(::typeof(kron), ::Type{Arg{1}}, p, Y::A, Ȳ::A, A::A, B::A)
    return ∇(zeroslike(A), kron, Arg{1}, p, Y, Ȳ, A, B)
end
function ∇(::typeof(kron), ::Type{Arg{2}}, p, Y::A, Ȳ::A, A::A, B::A)
    return ∇(zeroslike(B), kron, Arg{2}, p, Y, Ȳ, A, B)
end

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
