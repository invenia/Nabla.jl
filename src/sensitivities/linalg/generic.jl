# Implementation of sensitivities for unary linalg optimisations.
_ϵ, lb, ub = 3e-2, -3.0, 3.0
unary_linalg_optimisations = [
    (:-,          ∇Array,  ∇Array,  :(map(-, Ȳ)),                        (lb, ub)),
    (:trace,      ∇Array,  ∇Scalar, :(Diagonal(fill!(similar(X), Ȳ))),   (lb, ub)),
    (:inv,        ∇Array,  ∇Array,  :(-transpose(Y) * Ȳ * transpose(Y)), (lb, ub)),
    (:det,        ∇Array,  ∇Scalar, :(Y * Ȳ * transpose(inv(X))),        (_ϵ, ub)),
    (:logdet,     ∇Array,  ∇Scalar, :(Ȳ * transpose(inv(X))),            (_ϵ, ub)),
    (:transpose,  ∇Array,  ∇Array,  :(transpose(Ȳ)),                     (lb, ub)),
    (:ctranspose, ∇Array,  ∇Array,  :(ctranspose(Ȳ)),                    (lb, ub)),
    (:vecnorm,    ∇Array,  ∇Scalar, :(Ȳ ./ Y .* abs2.(X) ./ X),          (lb, ub)),
    (:vecnorm,    ∇Scalar, ∇Scalar, :(Ȳ * sign(X)),                      (lb, ub))
]
for (f, T_In, T_Out, X̄, bounds) in unary_linalg_optimisations
    @eval import Base.$f
    @eval @explicit_intercepts $f Tuple{$T_In}
    @eval ∇(::typeof($f), ::Type{Arg{1}}, p, Y::$T_Out, Ȳ::$T_Out, X::$T_In) = $X̄
end

# Implementation of sensitivities for binary linalg optimisations.
const A = ∇Array
const S = ∇Scalar
const AS = Union{∇Scalar, ∇Array}
δ = 1e-5
binary_linalg_optimisations = [
    (:*,          A, A, AS,
        :(A_mul_Bc(Ȳ, B)),
        :(Ac_mul_B(A, Ȳ))),
    (:At_mul_B,   A, A, AS,
        :(A_mul_Bt(B, Ȳ)),
        :(getfield(Base, :*)(A, Ȳ))),
    (:A_mul_Bt,   A, A, AS,
        :(getfield(Base, :*)(Ȳ, B)),
        :(At_mul_B(Ȳ, A))),
    (:At_mul_Bt,  A, A, AS,
        :(At_mul_Bt(B, Ȳ)),
        :(At_mul_Bt(Ȳ, A))),
    (:Ac_mul_B,   A, A, AS,
        :(A_mul_Bt(B, Ȳ)),
        :(getfield(Base, :*)(A, Ȳ))),
    (:A_mul_Bc,   A, A, AS,
        :(getfield(Base, :*)(Ȳ, B)),
        :(Ac_mul_B(Ȳ, A))),
    (:Ac_mul_Bc,  A, A, AS,
        :(Ac_mul_Bc(B, Ȳ)),
        :(Ac_mul_Bc(Ȳ, A))),
    (:/,          A, A, AS,
        :(A_rdiv_Bt(Ȳ, B)),
        :(-At_mul_B(Y, A_rdiv_Bt(Ȳ, B)))),
    (:At_rdiv_B,  A, A, AS,
        :(A_ldiv_Bt(B, Ȳ)),
        :(-At_mul_B(Y, A_rdiv_Bt(Ȳ, B)))),
    (:A_rdiv_Bt,  A, A, AS,
        :(getfield(Base, :/)(Ȳ, B)),
        :(-At_ldiv_Bt(B, Ȳ) * Y)),
    (:At_rdiv_Bt, A, A, AS,
        :(At_ldiv_Bt(B, Ȳ)),
        :(-At_ldiv_Bt(B, Ȳ) * Y)),
    (:Ac_rdiv_B,  A, A, AS,
        :(A_ldiv_Bc(B, Ȳ)),
        :(-Ac_mul_B(Y, A_rdiv_Bc(Ȳ, B)))),
    (:A_rdiv_Bc,  A, A, AS,
        :(getfield(Base, :/)(Ȳ, B)),
        :(-At_ldiv_Bt(B, Ȳ) * Y)),
    (:Ac_rdiv_Bc, A, A, AS,
        :(Ac_ldiv_Bc(B, Ȳ)),
        :(-Ac_ldiv_Bc(B, Ȳ) * Y)),
    (:\,          A, A, AS,
        :(-A_mul_Bt(At_ldiv_B(A, Ȳ), Y)),
        :(At_ldiv_B(A, Ȳ))),
    (:At_ldiv_B,  A, A, AS,
        :(-A_mul_Bt(Y, getfield(Base, :\)(A, Ȳ))),
        :(getfield(Base, :\)(A, Ȳ))),
    (:A_ldiv_Bt,  A, A, AS,
        :(-At_mul_Bt(At_rdiv_B(Ȳ, A), Y)),
        :(At_rdiv_B(Ȳ, A))),
    (:At_ldiv_Bt, A, A, AS,
        :(-Y * At_rdiv_Bt(Ȳ, A)),
        :(At_rdiv_Bt(Ȳ, A))),
    (:Ac_ldiv_B,  A, A, AS,
        :(-A_mul_Bc(Y, getfield(Base, :\)(A, Ȳ))),
        :(getfield(Base, :\)(A, Ȳ))),
    (:A_ldiv_Bc,  A, A, AS,
        :(-Ac_mul_Bc(Ac_rdiv_B(Ȳ, A), Y)),
        :(Ac_rdiv_B(Ȳ, A))),
    (:Ac_ldiv_Bc, A, A, AS,
        :(-Y * Ac_rdiv_Bc(Ȳ, A)),
        :(Ac_rdiv_Bc(Ȳ, A))),
    (:vecnorm,    A, S, S,
        :(Ȳ .* Y^(1 - B) .* abs.(A).^B ./ A),
        :(Ȳ * (Y^(1 - B) * sum(abs.(A).^B .* log.(abs.(A))) - Y * log(Y)) / B)),
    (:vecnorm,    S, S, S,
        :(Ȳ * sign(A)),
        :(0)),

]
for (f, T_A, T_B, T_Y, Ā, B̄) in binary_linalg_optimisations
    @eval import Base.$f
    @eval @explicit_intercepts $f Tuple{$T_A, $T_B}
    @eval ∇(::typeof($f), ::Type{Arg{1}}, p, Y::$T_Y, Ȳ::$T_Y, A::$T_A, B::$T_B) = $Ā
    @eval ∇(::typeof($f), ::Type{Arg{2}}, p, Y::$T_Y, Ȳ::$T_Y, A::$T_A, B::$T_B) = $B̄
end

# Sensitivities for the Kronecker product:
import Base.kron
@explicit_intercepts kron Tuple{A, A}

# The allocating versions simply allocate and then call the in-place versions.
∇(::typeof(kron), ::Type{Arg{1}}, p, Y::A, Ȳ::A, A::A, B::A) =
    ∇(zeros(A), kron, Arg{1}, p, Y, Ȳ, A, B)
∇(::typeof(kron), ::Type{Arg{2}}, p, Y::A, Ȳ::A, A::A, B::A) =
    ∇(zeros(B), kron, Arg{2}, p, Y, Ȳ, A, B)

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
