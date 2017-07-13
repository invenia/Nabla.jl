# Implementation of sensitivities for unary linalg optimisations.
unary_linalg_optimisations = [
    (:-,          RealArray, RealArray, :(Base.map(-, Ȳ)),                      (lb, ub)),
    (:trace,      RealArray, Real,      :(Base.Diagonal(ones(size(X, 1)))),     (lb, ub)),
    (:inv,        RealArray, RealArray, :(-Base.transpose(Y) * Ȳ * Base.transpose(Y)), (lb, ub)),
    (:det,        RealArray, Real,      :(Y * Ȳ * Base.transpose(Base.inv(X))), (lb, ub)),
    (:logdet,     RealArray, Real,      :(Ȳ * Base.transpose(Base.inv(X))),     (_ϵ, ub)),
    (:transpose,  RealArray, RealArray, :(Base.transpose(Ȳ)),                   (lb, ub)),
    (:ctranspose, RealArray, RealArray, :(Base.ctranspose(Ȳ)),                  (lb, ub)),
]
for (f, T_In, T_Out, X̄, bounds) in unary_linalg_optimisations
    eval(DiffBase, add_intercept(f, :(Base.$f), :(Tuple{$(quot(T_In))})))
    @eval DiffBase ∇(::typeof($f), ::Type{Arg{1}}, p, Y::$T_Out, Ȳ::$T_Out, X::$T_In) = $X̄
    @eval DiffBase export $f
end

# Implementation of sensitivities for binary linalg optimisations.
const RA = RealArray
δ = 1e-5
binary_linalg_optimisations = [
    (:*,          RA, RA, RA,
        :(Base.A_mul_Bc(Ȳ, B)),
        :(Base.Ac_mul_B(A, Ȳ)),
        (f, A, B, Ā, B̄)->discrepancy(f, (A, B), δ)),
    (:At_mul_B,   RA, RA, RA,
        :(Base.A_mul_Bt(B, Ȳ)),
        :(getfield(Base, :*)(A, Ȳ)),
        (f, A, B, Ā, B̄)->discrepancy(f, (A, B), δ)),
    (:A_mul_Bt,   RA, RA, RA,
        :(getfield(Base, :*)(Ȳ, B)),
        :(Base.At_mul_B(Ȳ, A)),
        (f, A, B, Ā, B̄)->discrepancy(f, (A, B), δ)),
    (:At_mul_Bt,  RA, RA, RA,
        :(Base.At_mul_Bt(B, Ȳ)),
        :(Base.At_mul_Bt(Ȳ, A)),
        (f, A, B, Ā, B̄)->discrepancy(f, (A, B), δ)),
    (:Ac_mul_B,   RA, RA, RA,
        :(Base.A_mul_Bt(B, Ȳ)),
        :(getfield(Base, :*)(A, Ȳ)),
        (f, A, B, Ā, B̄)->discrepancy(f, (A, B), δ)),
    (:A_mul_Bc,   RA, RA, RA,
        :(getfield(Base, :*)(Ȳ, B)),
        :(Base.Ac_mul_B(Ȳ, A)),
        (f, A, B, Ā, B̄)->discrepancy(f, (A, B), δ)),
    (:Ac_mul_Bc,  RA, RA, RA,
        :(Base.Ac_mul_Bc(B, Ȳ)),
        :(Base.Ac_mul_Bc(Ȳ, A)),
        (f, A, B, Ā, B̄)->discrepancy(f, (A, B), δ)),
    (:/,          RA, RA, RA,
        :(Base.A_rdiv_Bt(Ȳ, B)),
        :(-Base.At_mul_B(Y, Base.A_rdiv_Bt(Ȳ, B))),
        (f, A, B, Ā, B̄)->compute_errs.((∇(A * inv(B))[A], ∇(A * inv(B))[B]), (Ā, B̄))),
    (:At_rdiv_B,  RA, RA, RA,
        :(Base.A_ldiv_Bt(B, Ȳ)),
        :(-Base.At_mul_B(Y, Base.A_rdiv_Bt(Ȳ, B))),
        (f, A, B, Ā, B̄)->compute_errs.((∇(A.' * inv(B))[A], ∇(A.' * inv(B))[B]), (Ā, B̄))),
    (:A_rdiv_Bt,  RA, RA, RA,
        :(getfield(Base, :/)(Ȳ, B)),
        :(-Base.At_ldiv_Bt(B, Ȳ) * Y),
        (f, A, B, Ā, B̄)->compute_errs.((∇(A * inv(B.'))[A], ∇(A * inv(B.'))[B]), (Ā, B̄))),
    (:At_rdiv_Bt, RA, RA, RA,
        :(Base.At_ldiv_Bt(B, Ȳ)),
        :(-Base.At_ldiv_Bt(B, Ȳ) * Y),
        (f, A, B, Ā, B̄)->compute_errs.((∇(A.' * inv(B).')[A], ∇(A.' * inv(B).')[B]), (Ā, B̄))),
    (:Ac_rdiv_B,  RA, RA, RA,
        :(Base.A_ldiv_Bc(B, Ȳ)),
        :(-Base.Ac_mul_B(Y, Base.A_rdiv_Bc(Ȳ, B))),
        (f, A, B, Ā, B̄)->compute_errs.((∇(A' * inv(B))[A], ∇(A' * inv(B))[B]), (Ā, B̄))),
    (:A_rdiv_Bc,  RA, RA, RA,
        :(getfield(Base, :/)(Ȳ, B)),
        :(-Base.At_ldiv_Bt(B, Ȳ) * Y),
        (f, A, B, Ā, B̄)->compute_errs.((∇(A * (inv(B')))[A], ∇(A * (inv(B')))[B]), (Ā, B̄))),
    (:Ac_rdiv_Bc, RA, RA, RA,
        :(Base.Ac_ldiv_Bc(B, Ȳ)),
        :(-Base.Ac_ldiv_Bc(B, Ȳ) * Y),
        (f, A, B, Ā, B̄)->compute_errs.((∇(A' * inv(B)')[A], ∇(A' * inv(B)')[B]), (Ā, B̄))),
    (:\,          RA, RA, RA,
        :(-Base.A_mul_Bt(Base.At_ldiv_B(A, Ȳ), Y)),
        :(Base.At_ldiv_B(A, Ȳ)),
        (f, A, B, Ā, B̄)->compute_errs.((∇(inv(A) * B)[A], ∇(inv(A) * B)[B]), (Ā, B̄))),
    (:At_ldiv_B,  RA, RA, RA,
        :(-Base.A_mul_Bt(Y, getfield(Base, :\)(A, Ȳ))),
        :(getfield(Base, :\)(A, Ȳ)),
        (f, A, B, Ā, B̄)->compute_errs.((∇(inv(A.') * B)[A], ∇(inv(A.') * B)[B]), (Ā, B̄))),
    (:A_ldiv_Bt,  RA, RA, RA,
        :(-Base.At_mul_Bt(Base.At_rdiv_B(Ȳ, A), Y)),
        :(Base.At_rdiv_B(Ȳ, A)),
        (f, A, B, Ā, B̄)->compute_errs.((∇(inv(A) * B.')[A], ∇(inv(A) * B.')[B]), (Ā, B̄))),
    (:At_ldiv_Bt, RA, RA, RA,
        :(-Y * Base.At_rdiv_Bt(Ȳ, A)),
        :(Base.At_rdiv_Bt(Ȳ, A)),
        (f, A, B, Ā, B̄)->compute_errs.((∇(inv(A.') * B.')[A], ∇(inv(A.') * B.')[B]), (Ā, B̄))),
    (:Ac_ldiv_B,  RA, RA, RA,
        :(-Base.A_mul_Bc(Y, getfield(Base, :\)(A, Ȳ))),
        :(getfield(Base, :\)(A, Ȳ)),
        (f, A, B, Ā, B̄)->compute_errs.((∇(inv(A') * B)[A], ∇(inv(A') * B)[B]), (Ā, B̄))),
    (:A_ldiv_Bc,  RA, RA, RA,
        :(-Base.Ac_mul_Bc(Base.Ac_rdiv_B(Ȳ, A), Y)),
        :(Base.Ac_rdiv_B(Ȳ, A)),
        (f, A, B, Ā, B̄)->compute_errs.((∇(inv(A) * B')[A], ∇(inv(A) * B')[B]), (Ā, B̄))),
    (:Ac_ldiv_Bc, RA, RA, RA,
        :(-Y * Base.Ac_rdiv_Bc(Ȳ, A)),
        :(Base.Ac_rdiv_Bc(Ȳ, A)),
        (f, A, B, Ā, B̄)->compute_errs.((∇(inv(A') * B')[A], ∇(inv(A') * B')[B]), (Ā, B̄))),
]
for (f, T_A, T_B, T_Y, Ā, B̄, expected) in binary_linalg_optimisations

    # Create intercepts and export.
    accepted = :(Tuple{$(quot(T_A)), $(quot(T_B))})
    eval(DiffBase, add_intercept(f, :(Base.$f), accepted))
    @eval DiffBase export $f

    # Define sensitivities.
    f_tp = @eval typeof($f)
    @eval DiffBase ∇(::$f_tp, ::Type{Arg{1}}, p, Y::$T_Y, Ȳ::$T_Y, A::$T_A, B::$T_B) = $Ā
    @eval DiffBase ∇(::$f_tp, ::Type{Arg{2}}, p, Y::$T_Y, Ȳ::$T_Y, A::$T_A, B::$T_B) = $B̄
end
