@testset "Symmetric" begin
    let rng = MersenneTwister(123456), N = 100
        for _ in 1:10
            X, V, Ȳ = randn.(rng, [N, N, N], [N, N, N])
            @test check_errs(Symmetric, Ȳ, X, V)
        end
    end
end
