import Nabla: ∇Ctx
using Cassette: Box
@testset "Array" begin

    let
        ctx = ∇Ctx(1)
        @test size(Box(ctx, ones(1, 2, 3, 4), (5, (4, 3)))) == (1, 2, 3, 4)
        @test length(Box(ctx, 1:3, (5, (4, 3)))) == 3
    end

    let
        rng = MersenneTwister(123456)
        x = randn(2, 10)
        f1 = x̂->reshape(x̂, 5, 4)
        f2 = x̂->reshape(x̂, (5, 4))
        @test check_errs(f1, randn(rng, size(f1(x))), x, randn(rng, size(x)...))
        @test check_errs(f2, randn(rng, size(f2(x))), x, randn(rng, size(x)...))
    end
end
