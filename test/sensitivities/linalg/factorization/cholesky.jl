@testset "Cholesky" begin

    import Nabla: level2partition, level3partition
    let rng = MersenneTwister(123456), N = 5
        A = randn(rng, N, N)
        r, d, B2, c = level2partition(A, 4, false)
        R, D, B3, C = level3partition(A, 4, 4, false)
        @test all(r .== R')
        @test all(d .== D)
        @test B2[1] == B3[1]
        @test all(c .== C)

        # Check that level2partition with 'U' is consistent with 'L'.
        rᵀ, dᵀ, B2ᵀ, cᵀ = level2partition(transpose(A), 4, true)
        @test r == rᵀ
        @test d == dᵀ
        @test B2' == B2ᵀ
        @test c == cᵀ

        # Check that level3partition with 'U' is consistent with 'L'.
        R, D, B3, C = level3partition(A, 2, 4, false)
        Rᵀ, Dᵀ, B3ᵀ, Cᵀ = level3partition(transpose(A), 2, 4, true)
        @test transpose(R) == Rᵀ
        @test transpose(D) == Dᵀ
        @test transpose(B3) == B3ᵀ
        @test transpose(C) == Cᵀ
    end

    import Nabla: chol_unblocked_rev, chol_blocked_rev
    let rng = MersenneTwister(123456), N = 10
        A, Ā = Matrix.(LowerTriangular.(randn.(Ref(rng), [N, N], [N, N])))
        # NOTE: BLAS gets angry if we don't materialize the Transpose objects first
        B, B̄ = Matrix.(transpose.([A, Ā]))
        @test chol_unblocked_rev(Ā, A, false) ≈ chol_blocked_rev(Ā, A, 1, false)
        @test chol_unblocked_rev(Ā, A, false) ≈ chol_blocked_rev(Ā, A, 3, false)
        @test chol_unblocked_rev(Ā, A, false) ≈ chol_blocked_rev(Ā, A, 5, false)
        @test chol_unblocked_rev(Ā, A, false) ≈ chol_blocked_rev(Ā, A, 10, false)
        @test chol_unblocked_rev(Ā, A, false) ≈ transpose(chol_unblocked_rev(B̄, B, true))

        @test chol_unblocked_rev(B̄, B, true) ≈ chol_blocked_rev(B̄, B, 1, true)
        @test chol_unblocked_rev(B̄, B, true) ≈ chol_blocked_rev(B̄, B, 5, true)
        @test chol_unblocked_rev(B̄, B, true) ≈ chol_blocked_rev(B̄, B, 10, true)
    end

    # Check sensitivities for lower-triangular version.
    let rng = MersenneTwister(123456), N = 10
        for _ in 1:10
            B, VB = randn.(Ref(rng), [N, N], [N, N])
            A, VA = B'B + 1e-6I, VB'VB + 1e-6I
            Ū = UpperTriangular(randn(rng, N, N))
            @test check_errs(X->cholesky(X).U, Ū, A, 1e-2 .* VA)
        end
    end

    let
        X_ = Matrix{Float64}(I, 5, 5)
        X = Leaf(Tape(), X_)
        C = cholesky(X)
        @test C isa Branch{<:Cholesky}
        @test getfield(C, :f) == LinearAlgebra.cholesky
        U = C.U
        @test U isa Branch{<:UpperTriangular}
        @test getfield(U, :f) == Base.getproperty
        @test unbox(U) ≈ cholesky(X_).U

        @test_throws ArgumentError ∇(X->cholesky(X).info)(X_)
    end
end
