@testset "sensitivities/blas" begin

let ϵ_abs = 1e-5, ϵ_rel = 1e-4, δ = 1e-6

    # Testing allocating sensitivities for dot.
    import Base.BLAS.dot
    let rng = MersenneTwister(123456)
        for _ in 1:10
            x, y, vx, vy = randn.(rng, [5, 5, 5, 5])
            δ_abs_x, δ_rel_x = check_Dv(dot, randn(rng), (x, y), δ .* (vx, vy))
            @test δ_abs_x < ϵ_abs && δ_rel_x < ϵ_rel
        end
    end
    let rng = MersenneTwister(123456)
        for _ in 1:10
            x, y, vx, vy = randn.(rng, [10, 6, 10, 6])
            _dot = (x, y)->dot(5, x, 2, y, 1)
            δ_abs_x, δ_rel_x = check_Dv(_dot, randn(rng), (x, y), δ .* (vx, vy))
            @test δ_abs_x < ϵ_abs && δ_rel_x < ϵ_rel
        end
    end

    # Testing allocating sensitivities for nrm2.
    import Base.BLAS.nrm2
    let rng = MersenneTwister(123456)
        for _ in 1:10
            δ_abs, δ_rel = check_Dv(nrm2, randn(rng), randn(rng, 100), δ * randn(rng, 100))
            @test δ_abs < ϵ_abs && δ_rel < ϵ_rel
        end
    end
    let rng = MersenneTwister(123456)
        λ = x->nrm2(50, x, 2)
        for _ in 1:10
            δ_abs, δ_rel = check_Dv(λ, randn(rng), randn(rng, 100), δ * randn(rng, 100))
            @test δ_abs < ϵ_abs && δ_rel < ϵ_rel
        end
    end

    # Testing allocating sensivities for asum.
    import Base.BLAS.asum
    let rng = MersenneTwister(123456)
        for _ in 1:10
            δ_abs, δ_rel = check_Dv(asum, randn(rng), randn(rng, 100), δ * randn(rng, 100))
            @test δ_abs < ϵ_abs && δ_rel < ϵ_rel
        end
        λ = x->asum(50, x, 2)
        for _ in 1:10
            δ_abs, δ_rel = check_Dv(λ, randn(rng), randn(rng, 100), δ * randn(rng, 100))
            @test δ_abs < ϵ_abs && δ_rel < ϵ_rel
        end
    end

    # # Testing for scal.
    # let x = randn(10)
    #     δ_abs, δ_rel = discrepancy(BLAS.scal, (5, 2.5, x, 2), δ, [false, true, true, false])
    #     @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
    # end

    # Test each of the four permutations of `gemm`.
    import Base.BLAS.gemm
    let rng = MersenneTwister(123456), N = 100, δ = 1e-6
        for _ in 1:10
            α, vα = randn.([rng, rng])
            A, B, VA, VB = randn.(rng, [N, N, N, N], [N, N, N, N])
            λs = [(α, A, B)->gemm('n', 'N', α, A, B),
                  (α, A, B)->gemm('T', 'n', α, A, B),
                  (α, A, B)->gemm('N', 't', α, A, B),
                  (α, A, B)->gemm('t', 'T', α, A, B)]
            γs = [(A, B)->gemm('n', 'N', A, B),
                  (A, B)->gemm('T', 'n', A, B),
                  (A, B)->gemm('N', 't', A, B),
                  (A, B)->gemm('t', 'T', A, B)]
            for (λ, γ) in zip(λs, γs)
                δ_abs_λ, δ_rel_λ = check_Dv(λ, λ(α, A, B), (α, A, B), δ .* (vα, VA, VB))
                @test δ_abs_λ < ϵ_abs && δ_rel_λ < ϵ_rel
                δ_abs_γ, δ_rel_γ = check_Dv(γ, γ(A, B), (A, B), δ .* (VA, VB))
                @test δ_abs_γ < ϵ_abs && δ_rel_γ < ϵ_rel
            end
        end
    end

    # Test both permutations of `gemv`.
    import Base.BLAS.gemv
    let rng = MersenneTwister(123456), N = 100, δ = 1e-6
        for _ in 1:10
            α, vα = randn.([rng, rng])
            A, VA = randn.(rng, [N, N], [N, N])
            x, vx = randn.(rng, [N, N])
            λs = [(α, A, x)->gemv('T', α, A, x), (α, A, x)->gemv('N', α, A, x)]
            γs = [(A, x)->gemv('T', A, x), (A, x)->gemv('N', A, x)]
            for (λ, γ) in zip(λs, γs)
                δ_abs_λ, δ_rel_λ = check_Dv(λ, λ(α, A, x), (α, A, x), δ .* (vα, VA, vx))
                @test δ_abs_λ < ϵ_abs && δ_rel_λ < ϵ_rel
                δ_abs_γ, δ_rel_γ = check_Dv(γ, γ(A, x), (A, x), δ .* (VA, vx))
                @test δ_abs_γ < ϵ_abs && δ_rel_γ < ϵ_rel
            end
        end
    end

    # # Test all four permutations of `syrk`.
    # import Base.BLAS.syrk
    # let rng = MersenneTwister(123456), N = 100, δ = 1e-6
    #     lmask, umask = full(LowerTriangular(ones(N, N))), full(UpperTriangular(ones(N, N)))
    #     for uplo in ['L', 'U'], trans in ['N', 'T']
    #         λ = (α, A)->(uplo == 'L' ? lmask : umask) .* syrk(uplo, trans, α, A)
    #         γ = A->(uplo == 'L' ? lmask : umask) .* syrk(uplo, trans, A)
    #         for _ in 1:10
    #             α, vα = randn.([rng, rng]) + [5.0, 0.0]
    #             α, vα = 1.0, randn(rng)
    #             A, VA = randn.(rng, [N, N], [N, N])
    #             δ_abs_λ, δ_rel_λ = check_Dv(λ, λ(α, A), (α, A), δ .* (vα, VA))
    #             @test δ_abs_λ < ϵ_abs && δ_rel_λ < ϵ_rel
    #             δ_abs_γ, δ_rel_γ = check_Dv(γ, γ(A), A, δ .* VA)
    #             @test δ_abs_γ < ϵ_abs && δ_rel_γ < ϵ_rel
    #         end
    #     end
    # end

    # Test all four permutations of `symm`.
    import Base.BLAS.symm
    let rng = MersenneTwister(123456), N = 10, δ = 1e-6
        lmask, umask = full(LowerTriangular(ones(N, N))), full(UpperTriangular(ones(N, N)))
        λs = [(α, A, B)->symm('L', 'L', α, A, B),
              (α, A, B)->symm('R', 'U', α, A, B),
              (α, A, B)->symm('R', 'L', α, A, B),
              (α, A, B)->symm('L', 'U', α, A, B)]
        γs = [(A, B)->symm('L', 'L', A, B),
              (A, B)->symm('R', 'U', A, B),
              (A, B)->symm('R', 'L', A, B),
              (A, B)->symm('L', 'U', A, B)]
        for _ in 1:10
            α, vα = randn.([rng, rng])
            A, B, VA, VB = randn.(rng, [N, N, N, N], [N, N, N, N])
            for (λ, γ) in zip(λs, γs)
                δ_abs_λ, δ_rel_λ = check_Dv(λ, λ(α, A, B), (α, A, B), δ .* (vα, VA, VB))
                @test δ_abs_λ < ϵ_abs && δ_rel_λ < ϵ_rel
                δ_abs_γ, δ_rel_γ = check_Dv(γ, γ(A, B), (A, B), δ .* (VA, VB))
                @test δ_abs_γ < ϵ_abs && δ_rel_γ < ϵ_rel
            end
        end
    end

    import Base.BLAS.symv
    let rng = MersenneTwister(123456), N = 10, δ = 1e-6
        λs = [(α, A, x)->symv('L', α, A, x), (α, A, x)->symv('U', α, A, x)]
        γs = [(A, x)->symv('L', A, x), (A, x)->symv('U', A, x)]
        for _ in 1:10
            α, vα = randn.([rng, rng])
            A, VA = randn.(rng, [N, N], [N, N])
            x, vx = randn.(rng, [N, N])
            for (λ, γ) in zip(λs, γs)
                δ_abs_λ, δ_rel_λ = check_Dv(λ, λ(α, A, x), (α, A, x), δ .* (vα, VA, vx))
                @test δ_abs_λ < ϵ_abs && δ_rel_λ < ϵ_rel
                δ_abs_γ, δ_rel_γ = check_Dv(γ, γ(A, x), (A, x), δ .* (VA, vx))
                @test δ_abs_γ < ϵ_abs && δ_rel_γ < ϵ_rel
            end
        end
    end

    import Base.BLAS.trmm
    let rng = MersenneTwister(123456), N = 10, δ = 1e-6
        for side in ['L', 'R'], ul in ['L', 'U'], tA in ['N', 'T'], dA in ['N']
            λ = (α, A, B)->trmm(side, ul, tA, dA, α, A, B)
            for _ in 1:10
                α, vα = randn.([rng, rng])
                A, B, VA, VB = randn.(rng, [N, N, N, N], [N, N, N, N])
                δ_abs, δ_rel = check_Dv(λ, λ(α, A, B), (α, A, B), δ .* (vα, VA, VB))
                @test δ_abs < ϵ_abs && δ_rel < ϵ_rel
            end
        end
    end

    import Base.BLAS.trmv
    let rng = MersenneTwister(123456), N = 10, δ = 1e-6
        for ul in ['L', 'U'], tA in ['N', 'T'], dA in ['N']
            λ = (A, b)->trmv(ul, tA, dA, A, b)
            for _ in 1:10
                A, VA = randn.(rng, [N, N], [N, N])
                b, vb = randn.(rng, [N, N])
                δ_abs, δ_rel = check_Dv(λ, λ(A, b), (A, b), δ .* (VA, vb))
                @test δ_abs < ϵ_abs && δ_rel < ϵ_rel
            end
        end
    end

    import Base.BLAS.trsm
    let rng = MersenneTwister(123456), N = 10, δ = 1e-6
        for side in ['L', 'R'], ul in ['L', 'U'], tA in ['N', 'T'], dA in ['N']
            λ = (α, A, X)->trsm(side, ul, tA, dA, α, A, X)
            for _ in 1:10
                α, vα = randn.([rng, rng])
                A, X, VA, VX = randn.(rng, [N, N, N, N], [N, N, N, N])
                A = randn(rng, N, N) + UniformScaling(3)
                δ_abs, δ_rel = check_Dv(λ, λ(α, A, X), (α, A, X), δ .* (vα, VA, VX))
                @test δ_abs < ϵ_abs && δ_rel < ϵ_rel
            end
        end
    end

    import Base.BLAS.trsv
    let rng = MersenneTwister(123456), N = 10, δ = 1e-6
        for ul in ['L', 'U'], tA in ['N', 'T'], dA in ['N']
            λ = (A, x)->trsv(ul, tA, dA, A, x)
            for _ in 1:10
                A = randn(rng, N, N) + UniformScaling(1)
                A = A.'A
                VA = randn(rng, N, N)
                x, vx = randn.(rng, [N, N])
                δ_abs, δ_rel = check_Dv(λ, λ(A, x), (A, x), δ .* (VA, vx))
                @test δ_abs < ϵ_abs && δ_rel < ϵ_rel
            end
        end
    end

end # let

end # testset
