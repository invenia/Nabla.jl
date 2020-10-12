@testset "Strided" begin
    RS = StridedMatrix{<:∇Scalar}
    RST = Transpose{<:∇Scalar, RS}
    RSA = Adjoint{<:∇Scalar, RS}
    strided_matmul_combinations = (
        (RS,  RS,  'N', 'C', :Ȳ, :B, 'C', 'N', :A, :Ȳ),
        (RST, RS,  'N', 'T', :B, :Ȳ, 'N', 'N', :A, :Ȳ),
        (RS,  RST, 'N', 'N', :Ȳ, :B, 'T', 'N', :Ȳ, :A),
        (RST, RST, 'T', 'T', :B, :Ȳ, 'T', 'T', :Ȳ, :A),
        (RSA, RS,  'N', 'C', :B, :Ȳ, 'N', 'N', :A, :Ȳ),
        (RS,  RSA, 'N', 'N', :Ȳ, :B, 'C', 'N', :Ȳ, :A),
        (RSA, RSA, 'C', 'C', :B, :Ȳ, 'C', 'C', :Ȳ, :A),
    )
    # TODO: This test seems like it doesn't actually test the combinations.
    let rng = MersenneTwister(123456), N = 100
        # Test strided matrix-matrix multiplication sensitivities.
        for (TA, TB, tCA, tDA, CA, DA, tCB, tDB, CB, DB) in strided_matmul_combinations
            A, B, VA, VB = randn.(Ref(rng), [N, N, N, N], [N, N, N, N])
            @test check_errs(*, A * B, (A, B), (VA, VB))
        end
    end
end
