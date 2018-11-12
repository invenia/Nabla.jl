@testset "Array" begin

    # These are old tests that should be updated at some point.
    @test_broken size(Leaf(Tape(), ones(1, 2, 3, 4))) == (1, 2, 3, 4)
    @test_broken length(Leaf(Tape(), 1:3)) == 3

    let
        rng = MersenneTwister(123456)
        x = randn(rng, 2, 10)
        f1 = x̂ -> reshape(x̂, 5, 4)
        f2 = x̂ -> reshape(x̂, (5, 4))
        @test_broken check_errs(f1, f1(x), x, randn(size(x)...))
        @test_broken check_errs(f2, f2(x), x, randn(size(x)...))
    end

    let
        rng = MersenneTwister(123456)

        # hcat tests.
        a, b, c = rand(rng, 3, 2), rand(rng, 3), rand(rng, 3, 3)
        ā, b̄, c̄ = (2 * ones(3, 2), 3 * ones(3), 4 * ones(3, 3))
        @test_broken ∇((a, b, c)->sum(hcat(2*a, 3*b, 4*c)))(a,b,c) == (ā, b̄, c̄)

        # vcat tests.
        a, b, c = rand(rng, 2, 4), rand(rng, 1, 4), rand(rng, 3, 4)
        ā, b̄, c̄ = (2 * ones(2, 4), 3 * ones(1, 4), 4 * ones(3, 4))
        @test_broken ∇((a, b, c)->sum(vcat(2a, 3b, 4c)))(a, b, c) == (ā, b̄, c̄)
    end
end
