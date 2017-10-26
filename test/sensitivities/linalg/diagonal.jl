@testset "Diagonal" begin
    let rng = MersenneTwister(123456), N = 10
        λ_2 = x->diagm(x, 2)
        λ_m3 = x->diagm(x, -3)
        λ_0 = x->diagm(x, 0)
        λ_false = x->diagm(x, false)
        λ_true = x->diagm(x, true)
        for _ in 1:10

            # Test vector case.
            x, vx = randn.(rng, [N, N])
            @test check_errs(diagm, diagm(randn(rng, N)), x, vx)

            @test check_errs(λ_2, λ_2(randn(rng, N)), x, vx)
            @test check_errs(λ_m3, λ_m3(randn(rng, N)), x, vx)
            @test check_errs(λ_0, λ_0(randn(rng, N)), x, vx)
            @test check_errs(λ_false, λ_false(randn(rng, N)), x, vx)
            @test check_errs(λ_true, λ_true(randn(rng, N)), x, vx)

            # Test scalar case.
            x, vx = randn(rng), randn(rng)
            @test check_errs(diagm, diagm(randn(rng)), x, vx)
        end
    end
    let rng = MersenneTwister(123456), N = 10
        for _ in 1:10
            A = randn(rng, N, N)
            VA = randn(rng, N, N)
            @test check_errs(x -> diag(x), randn(rng, N), A, VA)

            # Check various diagonals.
            for k = -3:3
                @test check_errs(x -> diag(x, k), randn(rng, N - abs(k)), A, VA)
            end
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
