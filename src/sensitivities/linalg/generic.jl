# Implementation of sensitivities for unary linalg optimisations.
unary_linalg_optimisations = [
    (:-,          ∇RealArray, ∇RealArray, :(map(-, Ȳ)),                        (lb, ub)),
    (:trace,      ∇RealArray, ∇Real,      :(Diagonal(fill!(similar(X), Ȳ))),  (lb, ub)),
    (:inv,        ∇RealArray, ∇RealArray, :(-transpose(Y) * Ȳ * transpose(Y)), (lb, ub)),
    (:det,        ∇RealArray, ∇Real,      :(Y * Ȳ * transpose(inv(X))),        (_ϵ, ub)),
    (:logdet,     ∇RealArray, ∇Real,      :(Ȳ * transpose(inv(X))),            (_ϵ, ub)),
    (:transpose,  ∇RealArray, ∇RealArray, :(transpose(Ȳ)),                     (lb, ub)),
    (:ctranspose, ∇RealArray, ∇RealArray, :(ctranspose(Ȳ)),                    (lb, ub)),
    (:vecnorm,    ∇RealArray, ∇Real,      :(Ȳ ./ Y .* abs2.(X) ./ X),          (lb, ub)),
    (:vecnorm,    ∇Real,      ∇Real,      :(Ȳ * sign(X)),                      (lb, ub))
]
for (f, T_In, T_Out, X̄, bounds) in unary_linalg_optimisations
    @eval import Base.$f
    @eval @explicit_intercepts $f Tuple{$T_In}
    @eval ∇(::typeof($f), ::Type{Arg{1}}, p, Y::$T_Out, Ȳ::$T_Out, X::$T_In) = $X̄
end

# Implementation of sensitivities for binary linalg optimisations.
const RA = ∇RealArray
const RRA = Union{∇Real, ∇RealArray}
δ = 1e-5
binary_linalg_optimisations = [
    (:*,          RA, RA, RRA,
        :(A_mul_Bc(Ȳ, B)),
        :(Ac_mul_B(A, Ȳ))),
    (:At_mul_B,   RA, RA, RRA,
        :(A_mul_Bt(B, Ȳ)),
        :(getfield(Base, :*)(A, Ȳ))),
    (:A_mul_Bt,   RA, RA, RRA,
        :(getfield(Base, :*)(Ȳ, B)),
        :(At_mul_B(Ȳ, A))),
    (:At_mul_Bt,  RA, RA, RRA,
        :(At_mul_Bt(B, Ȳ)),
        :(At_mul_Bt(Ȳ, A))),
    (:Ac_mul_B,   RA, RA, RRA,
        :(A_mul_Bt(B, Ȳ)),
        :(getfield(Base, :*)(A, Ȳ))),
    (:A_mul_Bc,   RA, RA, RRA,
        :(getfield(Base, :*)(Ȳ, B)),
        :(Ac_mul_B(Ȳ, A))),
    (:Ac_mul_Bc,  RA, RA, RRA,
        :(Ac_mul_Bc(B, Ȳ)),
        :(Ac_mul_Bc(Ȳ, A))),
    (:/,          RA, RA, RRA,
        :(A_rdiv_Bt(Ȳ, B)),
        :(-At_mul_B(Y, A_rdiv_Bt(Ȳ, B)))),
    (:At_rdiv_B,  RA, RA, RRA,
        :(A_ldiv_Bt(B, Ȳ)),
        :(-At_mul_B(Y, A_rdiv_Bt(Ȳ, B)))),
    (:A_rdiv_Bt,  RA, RA, RRA,
        :(getfield(Base, :/)(Ȳ, B)),
        :(-At_ldiv_Bt(B, Ȳ) * Y)),
    (:At_rdiv_Bt, RA, RA, RRA,
        :(At_ldiv_Bt(B, Ȳ)),
        :(-At_ldiv_Bt(B, Ȳ) * Y)),
    (:Ac_rdiv_B,  RA, RA, RRA,
        :(A_ldiv_Bc(B, Ȳ)),
        :(-Ac_mul_B(Y, A_rdiv_Bc(Ȳ, B)))),
    (:A_rdiv_Bc,  RA, RA, RRA,
        :(getfield(Base, :/)(Ȳ, B)),
        :(-At_ldiv_Bt(B, Ȳ) * Y)),
    (:Ac_rdiv_Bc, RA, RA, RRA,
        :(Ac_ldiv_Bc(B, Ȳ)),
        :(-Ac_ldiv_Bc(B, Ȳ) * Y)),
    (:\,          RA, RA, RRA,
        :(-A_mul_Bt(At_ldiv_B(A, Ȳ), Y)),
        :(At_ldiv_B(A, Ȳ))),
    (:At_ldiv_B,  RA, RA, RRA,
        :(-A_mul_Bt(Y, getfield(Base, :\)(A, Ȳ))),
        :(getfield(Base, :\)(A, Ȳ))),
    (:A_ldiv_Bt,  RA, RA, RRA,
        :(-At_mul_Bt(At_rdiv_B(Ȳ, A), Y)),
        :(At_rdiv_B(Ȳ, A))),
    (:At_ldiv_Bt, RA, RA, RRA,
        :(-Y * At_rdiv_Bt(Ȳ, A)),
        :(At_rdiv_Bt(Ȳ, A))),
    (:Ac_ldiv_B,  RA, RA, RRA,
        :(-A_mul_Bc(Y, getfield(Base, :\)(A, Ȳ))),
        :(getfield(Base, :\)(A, Ȳ))),
    (:A_ldiv_Bc,  RA, RA, RRA,
        :(-Ac_mul_Bc(Ac_rdiv_B(Ȳ, A), Y)),
        :(Ac_rdiv_B(Ȳ, A))),
    (:Ac_ldiv_Bc, RA, RA, RRA,
        :(-Y * Ac_rdiv_Bc(Ȳ, A)),
        :(Ac_rdiv_Bc(Ȳ, A))),
    (:vecnorm,    RA, ∇Real, ∇Real,
        :(Ȳ .* Y^(1 - B) .* abs.(A).^B ./ A),
        :(Ȳ * (Y^(1 - B) * sum(abs.(A).^B .* log.(abs.(A))) - Y * log(Y)) / B)),
    (:vecnorm,    ∇Real, ∇Real, ∇Real,
        :(Ȳ * sign(A)),
        :(0)),

]
for (f, T_A, T_B, T_Y, Ā, B̄) in binary_linalg_optimisations
    @eval import Base.$f
    @eval @explicit_intercepts $f Tuple{$T_A, $T_B}
    @eval ∇(::typeof($f), ::Type{Arg{1}}, p, Y::$T_Y, Ȳ::$T_Y, A::$T_A, B::$T_B) = $Ā
    @eval ∇(::typeof($f), ::Type{Arg{2}}, p, Y::$T_Y, Ȳ::$T_Y, A::$T_A, B::$T_B) = $B̄
end
