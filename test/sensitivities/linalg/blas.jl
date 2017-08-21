@testset "sensitivities/blas" begin

let ϵ_abs = 1e-5, ϵ_rel = 1e-4, δ = 1e-6

    # Testing allocating sensitivities for dot.
    import Base.BLAS.dot
    let rng = MersenneTwister(123456)
        for _ in 1:10
            x, y = randn(rng, 5), randn(rng, 5)
            λx, λy = x->dot(x, y), y->dot(x, y)

            δ_abs_x, δ_rel_x = check_Dv(λx, randn(rng), x, δ * randn(rng, 5))
            @test δ_abs_x < ϵ_abs && δ_rel_x < ϵ_rel
            δ_abs_y, δ_rel_y = check_Dv(λy, randn(rng), y, δ * randn(rng, 5))
            @test δ_abs_y < ϵ_abs && δ_rel_y < ϵ_rel
        end
    end
    let rng = MersenneTwister(123456)
        for _ in 1:10
            x, y = randn(rng, 10), randn(rng, 6)
            λx, λy = x->dot(5, x, 2, y, 1), y->dot(5, x, 2, y, 1)

            δ_abs_x, δ_rel_x = check_Dv(λx, randn(rng), x, δ * randn(rng, 10))
            @test δ_abs_x < ϵ_abs && δ_rel_x < ϵ_rel
            δ_abs_y, δ_rel_y = check_Dv(λy, randn(rng), y, δ * randn(rng, 6))
            @test δ_abs_y < ϵ_abs && δ_rel_y < ϵ_rel
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

    import Base.BLAS.gemm
    let rng = MersenneTwister(123456)
        α, A, B = randn(rng), randn(rng, 100, 100), randn(rng, 100, 100)
        λα = α->gemm('N', 'n', α, A, B)
        λA, λB A->gemm('N', 'n', α, A, B), B->gemm('n', 'N', α, A, B)
    end

    # # Testing for each permutation of possible (real-valued) transpositions for gemm.
    # let α = randn(), A = randn(3, 3), B = randn(3, 3)
    #     diff = [false, false, true, true, true]
    #     δ_abs_1, δ_rel_1 = discrepancy(BLAS.gemm, ('N', 'n', α, A, B), δ, diff)
    #     δ_abs_2, δ_rel_2 = discrepancy(BLAS.gemm, ('t', 'n', α, A, B), δ, diff)
    #     δ_abs_3, δ_rel_3 = discrepancy(BLAS.gemm, ('n', 'T', α, A, B), δ, diff)
    #     δ_abs_4, δ_rel_4 = discrepancy(BLAS.gemm, ('T', 't', α, A, B), δ, diff)

    #     @test all(map(check_abs, δ_abs_1)) && all(map(check_rel, δ_rel_1))
    #     @test all(map(check_abs, δ_abs_2)) && all(map(check_rel, δ_rel_2))
    #     @test all(map(check_abs, δ_abs_3)) && all(map(check_rel, δ_rel_3))
    #     @test all(map(check_abs, δ_abs_4)) && all(map(check_rel, δ_rel_4))
    # end

    # # Testing for gemv with both options.
    # let α = randn(), A = randn(3, 3), x = randn(3)
    #     diff = [false, true, true, true]
    #     δ_abs_1, δ_rel_1 = discrepancy(BLAS.gemv, ('N', α, A, x), δ, diff)
    #     δ_abs_2, δ_rel_2 = discrepancy(BLAS.gemv, ('T', α, A, x), δ, diff)

    #     @test all(map(check_abs, δ_abs_1)) && all(map(check_rel, δ_rel_1))
    #     @test all(map(check_abs, δ_abs_2)) && all(map(check_rel, δ_rel_2))
    # end

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
