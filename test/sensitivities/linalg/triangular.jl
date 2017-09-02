@testset "Triangular" begin
    let rng = MersenneTwister(123456), N = 10
        for _ in 1:10
            A, VA, L = randn.(rng, [N, N, N], [N, N, N])
            @test check_errs(LowerTriangular, LowerTriangular(L), A, VA)
        end
    end
    let rng = MersenneTwister(123456), N = 10
        for _ in 1:10
            A, VA, U = randn.(rng, [N, N, N], [N, N, N])
            @test check_errs(UpperTriangular, UpperTriangular(U), A, VA)
        end
    end
    let rng = MersenneTwister(123456), N = 10
        for _ in 1:10
            A = LowerTriangular(randn(rng, N, N))
            VA = LowerTriangular(randn(rng, N, N))
            @test check_errs(det, 10.0, A, VA)
        end
    end
    let rng = MersenneTwister(123456), N = 3
        for _ in 1:10
            A = UpperTriangular(randn(rng, N, N))
            VA = UpperTriangular(randn(rng, N, N))
            @test check_errs(det, 10.0, A, VA)
        end
    end
    let rng = MersenneTwister(123456), N = 10
        for _ in 1:10
            A = LowerTriangular(exp.(randn(rng, N, N)))
            VA = LowerTriangular(randn(rng, N, N))
            @test check_errs(logdet, 10.0, A, VA)
        end
    end
    let rng = MersenneTwister(123456), N = 3
        for _ in 1:10
            A = UpperTriangular(exp.(randn(rng, N, N)))
            VA = UpperTriangular(randn(rng, N, N))
            @test check_errs(logdet, 10.0, A, VA)
        end
    end
end
