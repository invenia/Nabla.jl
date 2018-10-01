@testset "Diagonal" begin
    let rng = MersenneTwister(123456), N = 10
        λ_2 = x->diagm(2 => x)
        λ_m3 = x->diagm(-3 => x)
        λ_0 = x->diagm(0 => x)
        λ_false = x->diagm(false => x)
        λ_true = x->diagm(true => x)
        for _ in 1:10
            x, vx = randn.(rng, [N, N])
            @test check_errs(x->diagm(0 => x), diagm(0 => randn(rng, N)), x, vx)

            @test check_errs(λ_2, λ_2(randn(rng, N)), x, vx)
            @test check_errs(λ_m3, λ_m3(randn(rng, N)), x, vx)
            @test check_errs(λ_0, λ_0(randn(rng, N)), x, vx)
            @test check_errs(λ_false, λ_false(randn(rng, N)), x, vx)
            @test check_errs(λ_true, λ_true(randn(rng, N)), x, vx)
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
