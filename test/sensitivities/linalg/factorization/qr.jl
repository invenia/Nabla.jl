@testset "QR" begin
    @testset "Comparison with finite differencing" begin
        rng = MersenneTwister(123456)
        n = 5
        A = randn(rng, n, n)
        VA = randn(rng, n, n)
        @test check_errs(X->qr(X).Q, randn(rng, n, n), A, VA)
        @test check_errs(X->qr(X).R, randn(rng, n, n), A, VA)
    end

    @testset "Branch consistency" begin
        X_ = Matrix(1.0I, 5, 3)
        X = Leaf(Tape(), X_)
        F = qr(X)
        @test F isa Branch{<:LinearAlgebra.QRCompactWY}
        @test getfield(F, :f) == qr
        @test unbox(F.Q) ≈ Matrix(1.0I, 5, 5)
        @test unbox(F.R) ≈ Matrix(1.0I, 3, 3)
        # Destructuring via iteration
        Q, R = F
        @test Q isa Branch{<:LinearAlgebra.QRCompactWYQ}
        @test R isa Branch{<:Matrix}
    end

    @testset "Tape updating" begin
        t = Tape()
        X_ = Matrix(1.0I, 4, 4)
        X = Leaf(t, X_)
        Q, R = qr(X)
        Y = Q*R
        Z = Q*Y*R
        rt = ∇(Z, X_)
        @test rt[2] isa NamedTuple{(:Q,:R)}
        @test rt[2].Q ≈ Matrix(2.0I, 4, 4)
        @test rt[2].R ≈ Matrix(2.0I, 4, 4)
    end
end
