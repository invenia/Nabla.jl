@testset "SVD" begin
    @testset "Comparison with finite differencing" begin
        rng = MersenneTwister(12345)
        for n in [4, 6, 10], m in [3, 5, 10]
            k = min(n, m)
            A = randn(rng, n, m)
            VA = randn(rng, n, m)
            @test check_errs(X->svd(X).U, randn(rng, n, k), A, VA)
            @test check_errs(X->svd(X).S, randn(rng, k), A, VA)
            @test check_errs(X->svd(X).V, randn(rng, m, k), A, VA)
        end
    end

    @testset "Error conditions" begin
        rng = MersenneTwister(12345)
        A = randn(rng, 5, 3)
        V̄t = randn(rng, 3, 3)
        @test_throws ArgumentError check_errs(X->svd(X).Vt, V̄t, A, A)
        @test_throws ErrorException check_errs(X->svd(X).whoops, V̄t, A, A)
    end

    @testset "Branch consistency" begin
        X_ = Matrix{Float64}(I, 3, 5)
        X = Leaf(Tape(), X_)
        USV = svd(X)
        @test USV isa Branch{<:SVD}
        @test getfield(USV, :f) == svd
        @test unbox(USV.U) ≈ Matrix{Float64}(I, 3, 3)
        @test unbox(USV.S) ≈ ones(Float64, 3)
        @test unbox(USV.V) ≈ Matrix{Float64}(I, 5, 3)
    end

    @testset "Helper functions" begin
        rng = MersenneTwister(12345)
        X = randn(rng, 10, 10)
        Y = randn(rng, 10, 10)
        @test Nabla.mulsubtrans!(copy(X), Y) ≈ Y .* (X - X')
        @test Nabla.eyesubx!(copy(X)) ≈ I - X
        @test Nabla.add!(copy(X), Y) ≈ X + Y
    end
end
