import Base: *, At_mul_B, Ac_mul_B, A_mul_Bt, A_mul_Bc, At_mul_Bt, Ac_mul_Bc,
    \, /, At_ldiv_B, Ac_ldiv_B, A_ldiv_Bt, A_ldiv_Bc, At_ldiv_Bt, Ac_ldiv_Bc,
    At_rdiv_B, Ac_rdiv_B, A_rdiv_Bt, A_rdiv_Bc, At_rdiv_Bt, Ac_rdiv_Bc

matmul = [
    (:*, :(A_mul_Bc(Ȳ, B)), :(Ac_mul_B(A, Ȳ))),
    (:At_mul_B, :(A_mul_Bt(B, Ȳ)), :(A * Ȳ)),
    (:A_mul_Bt, :(Ȳ * B), :(At_mul_B(Ȳ, A))),
    (:At_mul_Bt, :(At_mul_Bt(B, Ȳ)), :(At_mul_Bt(Ȳ, A))),
    (:Ac_mul_B, :(A_mul_Bt(B, Ȳ)), :(A * Ȳ)),
    (:A_mul_Bc, :(Ȳ * B), :(Ac_mul_B(Ȳ, A))),
    (:Ac_mul_Bc, :(Ac_mul_Bc(B, Ȳ)), :(Ac_mul_Bc(Ȳ, A))),
]

# All of the reverse-mode sensitivities for operations of the form Y = A \ B.
ldiv = [
    (:\,          :(-A_mul_Bt(At_ldiv_B(A, Ȳ), Y)), :(At_ldiv_B(A, Ȳ))),
    (:At_ldiv_B,  :(-A_mul_Bt(Y, A \ Ȳ)),           :(A \ Ȳ)),
    (:A_ldiv_Bt,  :(-A_mul_Bt(At_ldiv_B(A, Ȳ), Y)), :(At_rdiv_B(Ȳ, A))),
    (:At_ldiv_Bt, :(-A_mul_Bt(Y, A \ Ȳ)),           :(At_rdiv_Bt(Ȳ, A))),
    (:Ac_ldiv_B,  :(-A_mul_Bc(Y, A \ Ȳ)),           :(A \ Ȳ)),
    (:A_ldiv_Bc,  :(-A_mul_Bc(Ac_ldiv_B(A, Ȳ), Y)), :(Ac_rdiv_B(Ȳ, A))),
    (:Ac_ldiv_Bc, :(-A_mul_Bc(Y, A \ Ȳ)),           :(Ac_rdiv_Bc(Ȳ, A))),
]

# All of the reverse-mode sensitivities for operations of the form Y = A / B.
rdiv = [
    (:/,          :(A_rdiv_Bt(Ȳ, B)),  :(-At_mul_B(Y, A_rdiv_Bt(Ȳ, B)))),
    (:At_rdiv_B,  :(A_ldiv_Bt(B, Ȳ)),  :(-At_mul_B(Y, A_rdiv_Bt(Ȳ, B)))),
    (:A_rdiv_Bt,  :(Ȳ / B),            :(-At_ldiv_Bt(B, Ȳ) * Y)),
    (:At_rdiv_Bt, :(At_ldiv_Bt(B, Ȳ)), :(-At_ldiv_Bt(B, Ȳ) * Y)),
    (:Ac_rdiv_B,  :(A_ldiv_Bc(B, Ȳ)),  :(-Ac_mul_B(Y, A_rdiv_Bc(Ȳ, B)))),
    (:A_rdiv_Bc,  :(Ȳ / B),            :(-Ac_ldiv_Bc(B, Ȳ) * Y)),
    (:Ac_rdiv_Bc, :(Ac_ldiv_Bc(B, Ȳ)), :(-Ac_ldiv_Bc(B, Ȳ) * Y)),
]

# Primitive definition and unit testing for ldiv operations.
for (f, Ā, B̄) in vcat(matmul, ldiv, rdiv)
    @eval @primitive $f{T, V <: AbstractArray}(A::T, B::V) Y Ȳ $Ā $B̄
end
