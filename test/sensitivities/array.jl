@testset "sensitivities/array" begin
    @test size(Leaf(Tape(), ones(1, 2, 3, 4))) == (1, 2, 3, 4)
    @test length(Leaf(Tape(), 1:3)) == 3

    let c_rel = 1e4, ε_abs = 1e-16, rng = MersenneTwister(123456)
        x = randn(2, 10)
        A = randn(4, 5)
        f = x̂ -> reshape(x̂, 5, 4) * A
        @test check_errs(f, f(x), (x,), (1e-6 * randn(size(x)...),), ε_abs, c_rel)
    end
end
