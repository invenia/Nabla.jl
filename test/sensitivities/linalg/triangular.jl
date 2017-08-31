@testset "sensitivities/linalg/triangular" begin
    let rng = MersenneTwister(123456), N = 10, ϵ_abs = 1e-9, δ = 1e-6, c_rel = 1e6
        for _ in 1:10
            A, VA, L = randn.(rng, [N, N, N], [N, N, N])
            @test check_errs(LowerTriangular, LowerTriangular(L), A, VA, ϵ_abs, c_rel)
        end
    end
    let rng = MersenneTwister(123456), N = 10, ϵ_abs = 1e-9, δ = 1e-6, c_rel = 1e6
        for _ in 1:10
            A, VA, U = randn.(rng, [N, N, N], [N, N, N])
            @test check_errs(UpperTriangular, UpperTriangular(U), A, VA, ϵ_abs, c_rel)
        end
    end
    let rng = MersenneTwister(123456), N = 10, ϵ_abs = 1e-9, δ = 1e-6, c_rel = 1e6
        for _ in 1:10
            A = LowerTriangular(randn(rng, N, N))
            VA = LowerTriangular(δ .* randn(rng, N, N))
            @test check_errs(det, 10.0, A, VA, ϵ_abs, c_rel)
        end
    end
    let rng = MersenneTwister(123456), N = 3, ϵ_abs = 1e-9, δ = 1e-6, c_rel = 1e6
        for _ in 1:10
            A = UpperTriangular(randn(rng, N, N))
            VA = UpperTriangular(δ .* randn(rng, N, N))
            @test check_errs(det, 10.0, A, VA, ϵ_abs, c_rel)
        end
    end
    let rng = MersenneTwister(123456), N = 10, ϵ_abs = 1e-9, δ = 1e-6, c_rel = 1e6
        for _ in 1:10
            A = LowerTriangular(exp.(randn(rng, N, N)))
            VA = LowerTriangular(δ .* randn(rng, N, N))
            @test check_errs(logdet, 10.0, A, VA, ϵ_abs, c_rel)
        end
    end
    let rng = MersenneTwister(123456), N = 3, ϵ_abs = 1e-9, δ = 1e-6, c_rel = 1e6
        for _ in 1:10
            A = UpperTriangular(exp.(randn(rng, N, N)))
            VA = UpperTriangular(δ .* randn(rng, N, N))
            @test check_errs(logdet, 10.0, A, VA, ϵ_abs, c_rel)
        end
    end
end
