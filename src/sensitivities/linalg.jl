import Base: *, At_mul_B, Ac_mul_B, A_mul_Bt, A_mul_Bc, At_mul_Bt, Ac_mul_Bc,
    \, /, At_ldiv_B, Ac_ldiv_B, A_ldiv_Bt, A_ldiv_Bc, At_ldiv_Bt, Ac_ldiv_Bc,
    At_rdiv_B, Ac_rdiv_B, A_rdiv_Bt, A_rdiv_Bc, At_rdiv_Bt, Ac_rdiv_Bc
import Base.BLAS: gemm, gemm!, gemv, gemv!, ger!

# For the most general AbstractArrays for which multiplication is defined, default to
# Julia's multiplcation routines as stuff can't be accelerated via BLAS.
generic_matmul = [
    (:*, :(A_mul_Bc(Ȳ, B)), :(Ac_mul_B(A, Ȳ))),
    (:At_mul_B, :(A_mul_Bt(B, Ȳ)), :(A * Ȳ)),
    (:A_mul_Bt, :(Ȳ * B), :(At_mul_B(Ȳ, A))),
    (:At_mul_Bt, :(At_mul_Bt(B, Ȳ)), :(At_mul_Bt(Ȳ, A))),
    (:Ac_mul_B, :(A_mul_Bt(B, Ȳ)), :(A * Ȳ)),
    (:A_mul_Bc, :(Ȳ * B), :(Ac_mul_B(Ȳ, A))),
    (:Ac_mul_Bc, :(Ac_mul_Bc(B, Ȳ)), :(Ac_mul_Bc(Ȳ, A))),
]
for (f, Āexpr, B̄expr) in generic_matmul
    generate_primitive(f, [:(T <: AbstractArray), :(V <: AbstractArray)],
        [:A, :B], [:Ā, :B̄], [:T, :V], [true, true], :Y, :Ȳ,
        [:(Ā = $Āexpr), :(B̄ = $B̄expr)], [:(Ā += $Āexpr), :(B̄ += $B̄expr)])
end

# Use BLAS.gemm for strided matrix-matrix multiplication sensitivites.
strided_matmul = [
    (:*,         'N', 'C', :Ȳ, :B, 'C', 'N', :A, :Ȳ),
    (:At_mul_B,  'N', 'T', :B, :Ȳ, 'N', 'N', :A, :Ȳ),
    (:A_mul_Bt,  'N', 'N', :Ȳ, :B, 'T', 'N', :Ȳ, :A),
    (:At_mul_Bt, 'T', 'T', :B, :Ȳ, 'T', 'T', :Ȳ, :A),
    (:Ac_mul_B,  'N', 'C', :B, :Ȳ, 'N', 'N', :A, :Ȳ),
    (:A_mul_Bc,  'N', 'N', :Ȳ, :B, 'C', 'N', :Ȳ, :A),
    (:Ac_mul_Bc, 'C', 'C', :B, :Ȳ, 'C', 'C', :Ȳ, :A),
]
for (f, tCA, tDA, CA, DA, tCB, tDB, CB, DB) in strided_matmul
    n_Ā, u_Ā = :(Ā = gemm($tCA, $tDA, $CA, $DA)), :(gemm!($tCA, $tDA, 1., $CA, $DA, 1., Ā))
    n_B̄, u_B̄ = :(B̄ = gemm($tCB, $tDB, $CB, $DB)), :(gemm!($tCB, $tDB, 1., $CB, $DB, 1., B̄))
    generate_primitive(f, [:(T <: StridedMatrix), :(V <: StridedMatrix)],
        [:A, :B], [:Ā, :B̄], [:T, :V], [true, true], :Y, :Ȳ, [n_Ā, n_B̄], [u_Ā, u_B̄])
end

# Not every permutation of transpositions makes sense for matrix-vector multiplication. This
# list just includes those which make sense.
strided_matvecmul = [
    (:*,         'C', :ȳ, :b, 'C'),
    (:At_mul_B,  'T', :b, :ȳ, 'N'),
    (:Ac_mul_B,  'C', :b, :ȳ, 'N'),
]
for (f, tdA, CA, dA, tCb) in strided_matvecmul
    n_Ā, u_Ā = tdA == 'C' ? :(Ā = $CA * $dA') : :(Ā = $CA * $dA.'), :(ger!(1., $CA, $dA, Ā))
    n_b̄, u_b̄ = :(b̄ = gemv($tCb, A, ȳ)), :(b̄ = gemv!($tCb, 1., A, ȳ, 1., b̄))
    generate_primitive(f, [:(T <: StridedMatrix), :(V <: StridedVector)],
        [:A, :b], [:Ā, :b̄], [:T, :V], [true, true], :y, :ȳ, [n_Ā, n_b̄], [u_Ā, u_b̄])
end

# All of the reverse-mode sensitivities for operations of the form Y = A \ B.
strided_ldiv = [
    (:\, :(C = At_ldiv_B(A, Ȳ)), 'N', 'T', :C, :Y),
    (:At_ldiv_B, :(C = A \ Ȳ), 'N', 'T', :Y, :C),
    (:A_ldiv_Bt, :(C = At_rdiv_B(Ȳ, A)), 'T', 'T', :C, :Y),
    (:At_ldiv_Bt, :(C = At_rdiv_Bt(Ȳ, A)), 'N', 'N', :Y, :C),
    (:Ac_ldiv_B, :(C = A \ Ȳ), 'N', 'C', :Y, :C),
    (:A_ldiv_Bc, :(C = Ac_rdiv_B(Ȳ, A)), 'C', 'C', :C, :Y),
    (:Ac_ldiv_Bc, :(C = Ac_rdiv_Bc(Ȳ, A)), 'N', 'N', :Y, :C),
]

# All of the reverse-mode sensitivities for operations of the form Y = A / B.
# rdiv = [
#     (:/,          :(A_rdiv_Bt(Ȳ, B)),  :(-At_mul_B(Y, A_rdiv_Bt(Ȳ, B)))),
#     (:At_rdiv_B,  :(A_ldiv_Bt(B, Ȳ)),  :(-At_mul_B(Y, A_rdiv_Bt(Ȳ, B)))),
#     (:A_rdiv_Bt,  :(Ȳ / B),            :(-At_ldiv_Bt(B, Ȳ) * Y)),
#     (:At_rdiv_Bt, :(At_ldiv_Bt(B, Ȳ)), :(-At_ldiv_Bt(B, Ȳ) * Y)),
#     (:Ac_rdiv_B,  :(A_ldiv_Bc(B, Ȳ)),  :(-Ac_mul_B(Y, A_rdiv_Bc(Ȳ, B)))),
#     (:A_rdiv_Bc,  :(Ȳ / B),            :(-Ac_ldiv_Bc(B, Ȳ) * Y)),
#     (:Ac_rdiv_Bc, :(Ac_ldiv_Bc(B, Ȳ)), :(-Ac_ldiv_Bc(B, Ȳ) * Y)),
# ]

# Iterate through primitive definitions and add methods for each.
for (f, C, tA, tB, arg1, arg2) in strided_ldiv
    new_Ā = :(Ā = gemm($tA, $tB, -1.0, $arg1, $arg2))
    update_Ā = :(gemm!($tA, $tB, -1.0, $arg1, $arg2, 1.0, Ā))
    new_B̄ = :(B̄ = C)
    update_B̄ = :(broadcast!((b̄, c)->b̄ + c, B̄, B̄, C))
    generate_primitive(f, [:(T <: Any), :(V <: Any)],
        [:A, :B], [:Ā, :B̄], [:T, :V], [true, true], :Y, :Ȳ,
        [new_Ā, new_B̄], [update_Ā, update_B̄], C)
end
