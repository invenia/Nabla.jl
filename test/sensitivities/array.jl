@testset "sensitivities/array" begin
    @test size(Leaf(Tape(), ones(1, 2, 3, 4))) == (1, 2, 3, 4)
    @test length(Leaf(Tape(), 1:3)) == 3

    let rng = MersenneTwister(123456)
        x = randn(2, 10)
        f1 = x̂ -> reshape(x̂, 5, 4)
        f2 = x̂ -> reshape(x̂, (5, 4))
        @test check_errs(f1, f1(x), x, randn(size(x)...))
        @test check_errs(f2, f2(x), x, randn(size(x)...))
    end
end
