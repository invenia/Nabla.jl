@testset "sensitivities/linalg/diagonal" begin
    let rng = MersenneTwister(123456), N = 10
        for _ in 1:10
            x, vx = randn.(rng, [N, N])
            @test check_errs(diagm, diagm(randn(rng, N)), x, vx)
        end
    end
    let rng = MersenneTwister(123456), N = 10
        for _ in 1:10
            A = randn(rng, N)
            VA = randn(rng, N)
            @test check_errs(Diagonal, Diagonal(randn(rng, N)), A, VA)
        end
    end
    let rng = MersenneTwister(123456), N = 10
        for _ in 1:10
            A = randn(rng, N, N)
            VA = randn(rng, N, N)
            @test check_errs(Diagonal, Diagonal(randn(rng, N)), A, VA)
        end
    end
    let rng = MersenneTwister(123456), N = 10
        for _ in 1:10
            A = Diagonal(randn(rng, N))
            VA = Diagonal(randn(rng, N))
            @test check_errs(det, 10.0, A, VA)
        end
    end
    let rng = MersenneTwister(123456), N = 10
        for _ in 1:10
            A = Diagonal(exp.(randn(rng, N)))
            VA = Diagonal(randn(rng, N))
            @test check_errs(logdet, 10.0, A, VA)
        end
    end
end
