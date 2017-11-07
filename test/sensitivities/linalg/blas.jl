@testset "BLAS" begin
    import Base.BLAS.dot
    let rng = MersenneTwister(123456)
        for _ in 1:10
            x, y, vx, vy = randn.(rng, [5, 5, 5, 5])
            @test check_errs(dot, dot(x ,y), (x, y), (vx, vy))
        end
    end
    let rng = MersenneTwister(123456)
        for _ in 1:10
            x, y, vx, vy = randn.(rng, [10, 6, 10, 6])
            _dot = (x, y)->dot(5, x, 2, y, 1)
            @test check_errs(_dot, _dot(x, y), (x, y), (vx, vy))
        end
    end

    import Base.BLAS.nrm2
    let rng = MersenneTwister(123456)
        for _ in 1:10
            x, vx = randn(rng, 100), randn(rng, 100)
            @test check_errs(nrm2, randn(rng), x, vx)
        end
    end
    let rng = MersenneTwister(123456)
        λ = x->nrm2(50, x, 2)
        for _ in 1:10
            x, vx = randn(rng, 100), randn(rng, 100)
            @test check_errs(λ, randn(rng), x, vx)
        end
    end

    import Base.BLAS.asum
    let rng = MersenneTwister(123456)
        λ = x->asum(50, x, 2)
        for _ in 1:10
            x, vx = randn(rng, 100), randn(rng, 100)
            @test check_errs(asum, randn(rng), x, vx)
            @test check_errs(λ, randn(rng), x, vx)
        end
    end

    # Test each of the four permutations of `gemm`.
    import Base.BLAS.gemm
    let rng = MersenneTwister(123456), N = 100
        for tA in ['T', 'N'], tB in ['T', 'N']
            λ, γ = (α, A, B)->gemm(tA, tB, α, A, B), (A, B)->gemm(tA, tB, A, B)
            for _ in 1:10
                α, vα = randn.([rng, rng])
                A, B, VA, VB = randn.(rng, [N, N, N, N], [N, N, N, N])
                @test check_errs(λ, λ(α, A, B), (α, A, B), (vα, VA, VB))
                @test check_errs(γ, γ(A, B), (A, B), (VA, VB))
            end
        end
    end

    # Test both permutations of `gemv`.
    import Base.BLAS.gemv
    let rng = MersenneTwister(123456), N = 100
        for tA in ['T', 'N']
            λ, γ = (α, A, x)->gemv('T', α, A, x), (A, x)->gemv('T', A, x)
            for _ in 1:10
                α, vα = randn.([rng, rng])
                A, VA = randn.(rng, [N, N], [N, N])
                x, vx = randn.(rng, [N, N])
                @test check_errs(λ, λ(α, A, x), (α, A, x), (vα, VA, vx))
                @test check_errs(γ, γ(A, x), (A, x), (VA, vx))
            end
        end
    end

    # Test all four permutations of `symm`.
    import Base.BLAS.symm
    let rng = MersenneTwister(123456), N = 100
        lmask, umask = full(LowerTriangular(ones(N, N))), full(UpperTriangular(ones(N, N)))
        for side in ['L', 'R'], ul in ['L', 'U']
            λ, γ = (α, A, B)->symm(side, ul, α, A, B), (A, B)->symm(side, ul, A, B)
            for _ in 1:10
                α, vα = randn.([rng, rng])
                A, B, VA, VB = randn.(rng, [N, N, N, N], [N, N, N, N])
                @test check_errs(λ, λ(α, A, B), (α, A, B), (vα, VA, VB))
                @test check_errs(γ, γ(A, B), (A, B), (VA, VB))
            end
        end
    end

    import Base.BLAS.symv
    let rng = MersenneTwister(123456), N = 100
        for ul in ['L', 'U']
            λ, γ = (α, A, x)->symv(ul, α, A, x), (A, x)->symv(ul, A, x)
            for _ in 1:10
                α, vα = randn.([rng, rng])
                A, VA = randn.(rng, [N, N], [N, N])
                x, vx = randn.(rng, [N, N])
                @test check_errs(λ, λ(α, A, x), (α, A, x), (vα, VA, vx))
                @test check_errs(γ, γ(A, x), (A, x), (VA, vx))
            end
        end
    end

    import Base.BLAS.trmm
    let rng = MersenneTwister(123456), N = 10
        for side in ['L', 'R'], ul in ['L', 'U'], tA in ['N', 'T'], dA in ['N']
            λ = (α, A, B)->trmm(side, ul, tA, dA, α, A, B)
            for _ in 1:10
                α, vα = randn.([rng, rng])
                A, B, VA, VB = randn.(rng, [N, N, N, N], [N, N, N, N])
                @test check_errs(λ, λ(α, A, B), (α, A, B), (vα, VA, VB))
            end
        end
    end

    import Base.BLAS.trmv
    let rng = MersenneTwister(123456), N = 10
        for ul in ['L', 'U'], tA in ['N', 'T'], dA in ['N']
            λ = (A, b)->trmv(ul, tA, dA, A, b)
            for _ in 1:10
                A, VA = randn.(rng, [N, N], [N, N])
                b, vb = randn.(rng, [N, N])
                @test check_errs(λ, λ(A, b), (A, b), (VA, vb))
            end
        end
    end

    import Base.BLAS.trsm
    let rng = MersenneTwister(123456), N = 10
        for side in ['L', 'R'], ul in ['L', 'U'], tA in ['N', 'T'], dA in ['N']
            λ = (α, A, X)->trsm(side, ul, tA, dA, α, A, X)
            for _ in 1:10
                α, vα = randn.([rng, rng])
                A, X, VA, VX = randn.(rng, [N, N, N, N], [N, N, N, N])
                A = randn(rng, N, N) + UniformScaling(3)
                @test check_errs(λ, λ(α, A, X), (α, A, X), (vα, VA, VX))
            end
        end
    end

    import Base.BLAS.trsv
    let rng = MersenneTwister(123456), N = 10
        for ul in ['L', 'U'], tA in ['N', 'T'], dA in ['N']
            λ = (A, x)->trsv(ul, tA, dA, A, x)
            for _ in 1:10
                A = randn(rng, N, N) + UniformScaling(1)
                A = A.'A
                VA = randn(rng, N, N)
                x, vx = randn.(rng, [N, N])
                @test check_errs(λ, λ(A, x), (A, x), (VA, vx))
            end
        end
    end
end
