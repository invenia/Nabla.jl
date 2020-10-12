# # Not every permutation of transpositions makes sense for matrix-vector multiplication. This
# # list just includes those which make sense.
# strided_matvecmul = [
#     (:*,         'C', :ȳ, :b, 'C'),
#     (:At_mul_B,  'T', :b, :ȳ, 'N'),
#     (:Ac_mul_B,  'C', :b, :ȳ, 'N'),
# ]
# for (f, tdA, CA, dA, tCb) in strided_matvecmul
#     n_Ā, u_Ā = tdA == 'C' ? :(Ā = $CA * $dA') : :(Ā = $CA * $dA'), :(ger!(1., $CA, $dA, Ā))
#     n_b̄, u_b̄ = :(b̄ = gemv($tCb, A, ȳ)), :(b̄ = gemv!($tCb, 1., A, ȳ, 1., b̄))
#     generate_primitive(f, [:(T <: StridedMatrix), :(V <: StridedVector)],
#         [:A, :b], [:Ā, :b̄], [:T, :V], [true, true], :y, :ȳ, [n_Ā, n_b̄], [u_Ā, u_b̄])
# end

# # Operations of the for Y = A \ B
# strided_ldiv = [
#     (:\, :(C = At_ldiv_B(A, Ȳ)), 'N', 'T', :C, :Y),
#     (:At_ldiv_B, :(C = A \ Ȳ), 'N', 'T', :Y, :C),
#     (:A_ldiv_Bt, :(C = At_rdiv_B(Ȳ, A)), 'T', 'T', :C, :Y),
#     (:At_ldiv_Bt, :(C = At_rdiv_Bt(Ȳ, A)), 'N', 'N', :Y, :C),
#     (:Ac_ldiv_B, :(C = A \ Ȳ), 'N', 'C', :Y, :C),
#     (:A_ldiv_Bc, :(C = Ac_rdiv_B(Ȳ, A)), 'C', 'C', :C, :Y),
#     (:Ac_ldiv_Bc, :(C = Ac_rdiv_Bc(Ȳ, A)), 'N', 'N', :Y, :C),
# ]

# # Iterate through primitive definitions and add methods for each.
# for (f, C, tA, tB, arg1, arg2) in strided_ldiv
#     new_Ā = :(Ā = gemm($tA, $tB, -1.0, $arg1, $arg2))
#     update_Ā = :(gemm!($tA, $tB, -1.0, $arg1, $arg2, 1.0, Ā))
#     new_B̄ = :(B̄ = C)
#     update_B̄ = :(broadcast!((b̄, c)->b̄ + c, B̄, B̄, C))
#     generate_primitive(f, [:(T <: Any), :(V <: Any)],
#         [:A, :B], [:Ā, :B̄], [:T, :V], [true, true], :Y, :Ȳ,
#         [new_Ā, new_B̄], [update_Ā, update_B̄], C)
# end
