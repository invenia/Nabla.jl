@testset "sensitivities/linalg/diagonal" begin
    let rng = MersenneTwister(123456), N = 10, ϵ_abs = 1e-9, δ = 1e-6, c_rel = 1e6
        for _ in 1:10
            x, vx = randn.(rng, [N, N])
            @test check_errs(diagm, diagm(randn(rng, N)), x, vx, ϵ_abs, c_rel)
        end
    end
    let rng = MersenneTwister(123456), N = 10, ϵ_abs = 1e-9, δ = 1e-6, c_rel = 1e6
        for _ in 1:10
            A = randn(rng, N)
            VA = δ .* randn(rng, N)
            @test check_errs(Diagonal, Diagonal(randn(rng, N)), A, VA, ϵ_abs, c_rel)
        end
    end
    let rng = MersenneTwister(123456), N = 10, ϵ_abs = 1e-9, δ = 1e-6, c_rel = 1e6
        for _ in 1:10
            A = randn(rng, N, N)
            VA = δ .* randn(rng, N, N)
            @test check_errs(Diagonal, Diagonal(randn(rng, N)), A, VA, ϵ_abs, c_rel)
        end
    end
    let rng = MersenneTwister(123456), N = 10, ϵ_abs = 1e-9, δ = 1e-6, c_rel = 1e6
        for _ in 1:10
            A = Diagonal(randn(rng, N))
            VA = Diagonal(δ .* randn(rng, N))
            @test check_errs(det, 10.0, A, VA, ϵ_abs, c_rel)
        end
    end
    let rng = MersenneTwister(123456), N = 10, ϵ_abs = 1e-9, δ = 1e-6, c_rel = 1e6
        for _ in 1:10
            A = Diagonal(exp.(randn(rng, N)))
            VA = Diagonal(δ .* randn(rng, N))
            @test check_errs(logdet, 10.0, A, VA, ϵ_abs, c_rel)
        end
    end
end
