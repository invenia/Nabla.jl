@testset "Cholesky" begin
    # Check sensitivities for lower-triangular version.
    let rng = MersenneTwister(123456), N = 10
        for _ in 1:10
            B, VB = randn.(Ref(rng), [N, N], [N, N])
            Ū = UpperTriangular(randn(rng, N, N))
            @test check_errs(B->cholesky(B'B + I).U, Ū, B, 1e-2 .* VB)
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

        @test_throws MethodError ∇(X->cholesky(X).info)(X_)
    end

    let
        X_ = Matrix{Float64}(I, 5, 5)
        X = Leaf(Tape(), X_)
        U = cholesky(X).U
        C = Cholesky(U, 'U', 0)
        @test C isa Branch{<:Cholesky}
        @test getfield(C, :f) == LinearAlgebra.Cholesky
        @test unbox(C) == Cholesky(UpperTriangular(X_), 'U', 0)
        # Ensure we can still directly extract the .U field
        UU = C.U
        @test UU isa Branch{<:UpperTriangular}
        # And access .L as well
        LL = C.L
        @test LL isa Branch{<:LowerTriangular}
        # Make sure that computing the Cholesky and already having the Cholesky
        # produce the same results
        expected = Matrix(0.5I, 5, 5)
        @test ∇(X->det(cholesky(X).U))(X_)[1] ≈ expected
        @test ∇(X->det(cholesky(X).L))(X_)[1] ≈ expected
        @test ∇(X->det(Cholesky(cholesky(X).U, :U, 0).U))(X_)[1] ≈ expected
        @test ∇(X->det(Cholesky(cholesky(X).L, 'L', 0).U))(X_)[1] ≈ expected
    end
end
