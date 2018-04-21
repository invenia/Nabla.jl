@testset "Indexing" begin

    # Vector tests.
    let
        rng, N = MersenneTwister(123456), 5
        
        # Single index tests.
        @test check_errs(x->getindex(x, 1), randn(rng), randn(rng, N), randn(rng, N))
        @test check_errs(x->getindex(x, 3), randn(rng), randn(rng, N), randn(rng, N))

        # Multi-index tests.
        @test check_errs(x->getindex(x, 2:3), randn(rng, 2), randn(rng, N), randn(rng, N))
        @test check_errs(x->getindex(x, 1:2), randn(rng, 2), randn(rng, N), randn(rng, N))
        @test check_errs(x->getindex(x, 1:2:3), randn(rng, 2), randn(rng, N), randn(rng, N))
    end

    # Matrix tests.
    let
        rng, P, Q = MersenneTwister(123456), 6, 3
        X, V = randn(rng, P, Q), randn(rng, P, Q)

        # Single index tests.
        @test check_errs(X->getindex(X, 1), randn(rng), X, V)
        @test check_errs(X->getindex(X, 2, 3), randn(rng), X, V)
        @test check_errs(X->getindex(X, P, Q), randn(rng), X, V)
        @test check_errs(X->getindex(X, 1, 1), randn(rng), X, V)

        # Multi-index tests.
        @test check_errs(X->getindex(X, :), randn(rng, P * Q), X, V)
        @test check_errs(X->getindex(X, :, 1), randn(rng, P), X, V)
        @test check_errs(X->getindex(X, :, 3), randn(rng, P), X, V)
        @test check_errs(X->getindex(X, :, Q), randn(rng, P), X, V)
        @test check_errs(X->getindex(X, 1, :), randn(rng, Q), X, V)
        @test check_errs(X->getindex(X, 3, :), randn(rng, Q), X, V)
        @test check_errs(X->getindex(X, P, :), randn(rng, Q), X, V)
        @test check_errs(X->getindex(X, 1:P, 1:Q), randn(rng, P, Q), X, V)
        @test check_errs(X->getindex(X, 2:P, 3:Q), randn(rng, P-1, Q-2), X, V)
    end
end
