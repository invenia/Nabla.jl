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

    # Test all four permutations of `syrk`.
    import Base.BLAS.syrk
    let rng = MersenneTwister(123456), N = 10, δ = 1e-6
        for _ in 1:1
            α, vα = randn.([rng, rng])
            A, VA = randn.(rng, [N, N], [N, N])
            λs = [(α, A)->syrk('L', 'N', α, A),
                  (α, A)->syrk('U', 'N', α, A),
                  (α, A)->syrk('L', 'T', α, A),
                  (α, A)->syrk('U', 'T', α, A)]
            γs = [A->syrk('L', 'N', A),
                  A->syrk('U', 'N', A),
                  A->syrk('L', 'T', A),
                  A->syrk('U', 'T', A)]
            for (λ, γ) in zip(λs, γs)
                println("approximate_Dv")
                println(Nabla.approximate_Dv(λ, λ(α, A), (α, A), δ .* (vα, VA)))
                println("compute_Dv")
                println(Nabla.compute_Dv(λ, λ(α, A), (α, A), δ .* (vα, VA)))
                δ_abs_λ, δ_rel_λ = check_Dv(λ, λ(α, A), (α, A), δ .* (vα, VA))
                println((δ_abs_λ, δ_rel_λ))
                @test δ_abs_λ < ϵ_abs && δ_rel_λ < ϵ_rel
                δ_abs_γ, δ_rel_γ = check_Dv(γ, γ(A), A, δ .* VA)
                # @test δ_abs_γ < ϵ_abs && δ_rel_γ < ϵ_rel
            end
        end
    end

    # # Testing for syrk with all four permutations of inputs.
    # let α = randn(), A = randn(3, 2)
    #     diff = [false, false, true, true]
    #     δ_abs_1, δ_rel_1 = discrepancy(BLAS.syrk, ('L', 'N', α, A), δ, diff, tril)
    #     δ_abs_2, δ_rel_2 = discrepancy(BLAS.syrk, ('U', 'N', α, A), δ, diff, triu)
    #     δ_abs_3, δ_rel_3 = discrepancy(BLAS.syrk, ('L', 'T', α, A), δ, diff, tril)
    #     δ_abs_4, δ_rel_4 = discrepancy(BLAS.syrk, ('U', 'T', α, A), δ, diff, triu)

    #     @test all(map(check_abs, δ_abs_1)) && all(map(check_rel, δ_rel_1))
    #     @test all(map(check_abs, δ_abs_2)) && all(map(check_rel, δ_rel_2))
    #     @test all(map(check_abs, δ_abs_3)) && all(map(check_rel, δ_rel_3))
    #     @test all(map(check_abs, δ_abs_4)) && all(map(check_rel, δ_rel_4))
    # end

    # # Testing for symm with all four permutations of inputs.
    # let α = randn(), A = full(Symmetric(randn(3, 3))), B1 = randn(3, 2), B2 = randn(2, 3)
    #     diff = [false, false, true, true, true]
    #     δ_abs_1, δ_rel_1 = discrepancy(BLAS.symm, ('L', 'L', α, A, B1), δ, diff)
    #     δ_abs_2, δ_rel_2 = discrepancy(BLAS.symm, ('L', 'U', α, A, B1), δ, diff)
    #     δ_abs_3, δ_rel_3 = discrepancy(BLAS.symm, ('R', 'L', α, A, B2), δ, diff)
    #     δ_abs_4, δ_rel_4 = discrepancy(BLAS.symm, ('R', 'U', α, A, B2), δ, diff)

    #     @test all(map(check_abs, δ_abs_1)) && all(map(check_rel, δ_rel_1))
    #     @test all(map(check_abs, δ_abs_2)) && all(map(check_rel, δ_rel_2))
    #     @test all(map(check_abs, δ_abs_3)) && all(map(check_rel, δ_rel_3))
    #     @test all(map(check_abs, δ_abs_4)) && all(map(check_rel, δ_rel_4))
    # end

    # # Testing for symv with both permutations.
    # let α = randn(), A = full(Symmetric(randn(3, 3))), B = randn(3)
    #     diff = [false, true, true, true]
    #     δ_abs_1, δ_rel_1 = discrepancy(BLAS.symv, ('L', α, A, B), δ, diff)
    #     δ_abs_2, δ_rel_2 = discrepancy(BLAS.symv, ('U', α, A, B), δ, diff)

    #     @test all(map(check_abs, δ_abs_1)) && all(map(check_rel, δ_rel_1))
    #     @test all(map(check_abs, δ_abs_2)) && all(map(check_rel, δ_rel_2))
    # end

    # # Testing for trmm.
    # let α = randn(), A = randn(3, 3), B = randn(3, 3)
    #     diff = [false, false, false, false, true, true, true]
    #     for side in ['L', 'R'], ul in ['L', 'U'], tA in ['N', 'T'], dA in ['N']
    #         δ_abs, δ_rel = discrepancy(BLAS.trmm, (side, ul, tA, dA, α, A, B), δ, diff)
    #         @test all(map(check_abs, δ_abs)) && all(map(check_rel, δ_rel))
    #     end
    # end

    # # Testing for trmv.
    # let A = randn(3, 3), B = randn(3)
    #     diff = [false, false, false, true, true]
    #     for ul in ['L', 'U'], tA in ['N', 'T'], dA in ['N']
    #         δ_abs, δ_rel = discrepancy(BLAS.trmv, (ul, tA, dA, A, B), δ, diff)
    #         @test all(map(check_abs, δ_abs)) && all(map(check_rel, δ_rel))
    #     end
    # end

    # # Testing for trsm.
    # let α = randn(), A = randn(3, 3), B = randn(3, 3)
    #     diff = [false, false, false, false, true, true, true]
    #     for side in ['L', 'R'], ul in ['L', 'U'], tA in ['N', 'T'], dA in ['N']
    #         δ_abs, δ_rel = discrepancy(BLAS.trsm, (side, ul, tA, dA, α, A, B), δ, diff)
    #         @test all(map(check_abs, δ_abs)) && all(map(check_rel, δ_rel))
    #     end
    # end

    # # Testing for trsv.
    # let A = randn(3, 3), B = randn(3)
    #     diff = [false, false, false, true, true]
    #     for ul in ['L', 'U'], tA in ['N', 'T'], dA in ['N']
    #         δ_abs, δ_rel = discrepancy(BLAS.trsv, (ul, tA, dA, A, B), 1e-8, diff)
    #         @test all(map(check_abs, δ_abs)) && all(map(check_rel, δ_rel))
    #     end
    # end

end # let

end # testset
