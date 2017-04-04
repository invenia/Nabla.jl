print("sensitivities/blas.jl... ")

let ϵ_abs = 1e-5, ϵ_rel = 1e-4, δ = 1e-6

    # Testing for dot.
    let x = randn(10), y = randn(6)
        δ_abs, δ_rel = discrepancy(dot, ((x[1:5]), (y[1:5])), δ)
        @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel) &&
              all(δ_abs[2] .< ϵ_abs) && all(δ_rel[2] .< ϵ_rel)

        diff = [false, true, false, true, false]
        δ_abs, δ_rel = discrepancy(dot, (5, x, 2, y, 1), δ, diff)
        @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel) &&
              all(δ_abs[2] .< ϵ_abs) && all(δ_rel[2] .< ϵ_rel)
    end

    # Testing for nrm2.
    let x = randn(10)
        δ_abs, δ_rel = discrepancy(BLAS.nrm2, (x,), δ)
        @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)

        δ_abs, δ_rel = discrepancy(BLAS.nrm2, (5, x, 2), δ, [false, true, false])
        @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
    end

    # Testing for asum.
    let x = randn(10)
        δ_abs, δ_rel = discrepancy(BLAS.asum, (x,), δ)
        @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)

        δ_abs, δ_rel = discrepancy(BLAS.asum, (5, x, 2), δ, [false, true, false])
        @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
    end

    # # Testing for scal.
    # let x = randn(10)
    #     δ_abs, δ_rel = discrepancy(BLAS.scal, (5, 2.5, x, 2), δ, [false, true, true, false])
    #     @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
    # end

    # Testing for each permutation of possible (real-valued) transpositions for gemm.
    let α = randn(), A = randn(3, 3), B = randn(3, 3)
        diff = [false, false, true, true, true]
        δ_abs_1, δ_rel_1 = discrepancy(BLAS.gemm, ('N', 'n', α, A, B), δ, diff)
        δ_abs_2, δ_rel_2 = discrepancy(BLAS.gemm, ('t', 'n', α, A, B), δ, diff)
        δ_abs_3, δ_rel_3 = discrepancy(BLAS.gemm, ('n', 'T', α, A, B), δ, diff)
        δ_abs_4, δ_rel_4 = discrepancy(BLAS.gemm, ('T', 't', α, A, B), δ, diff)

        check_abs(x) = all(x .< ϵ_abs)
        check_rel(x) = all(x .< ϵ_rel)
        @test all(map(check_abs, δ_abs_1)) && all(map(check_rel, δ_rel_1))
        @test all(map(check_abs, δ_abs_2)) && all(map(check_rel, δ_rel_2))
        @test all(map(check_abs, δ_abs_3)) && all(map(check_rel, δ_rel_3))
        @test all(map(check_abs, δ_abs_4)) && all(map(check_rel, δ_rel_4))
    end

    # Testing for gemv with both options.
    let α = randn(), A = randn(3, 3), x = randn(3)
        diff = [false, true, true, true]
        δ_abs_1, δ_rel_1 = discrepancy(BLAS.gemv, ('N', α, A, x), δ, diff)
        δ_abs_2, δ_rel_2 = discrepancy(BLAS.gemv, ('T', α, A, x), δ, diff)

        check_abs(x) = all(x .< ϵ_abs)
        check_rel(x) = all(x .< ϵ_rel)
        @test all(map(check_abs, δ_abs_1)) && all(map(check_rel, δ_rel_1))
        @test all(map(check_abs, δ_abs_2)) && all(map(check_rel, δ_rel_2))
    end

    # Testing for syrk with all four permutations of inputs.
    let α = randn(), A = randn(3, 2)
        diff = [false, false, true, true]
        δ_abs_1, δ_rel_1 = discrepancy(BLAS.syrk, ('L', 'N', α, A), δ, diff, tril)
        δ_abs_2, δ_rel_2 = discrepancy(BLAS.syrk, ('U', 'N', α, A), δ, diff, triu)
        δ_abs_3, δ_rel_3 = discrepancy(BLAS.syrk, ('L', 'T', α, A), δ, diff, tril)
        δ_abs_4, δ_rel_4 = discrepancy(BLAS.syrk, ('U', 'T', α, A), δ, diff, triu)

        check_abs(x) = all(x .< ϵ_abs)
        check_rel(x) = all(x .< ϵ_rel)
        @test all(map(check_abs, δ_abs_1)) && all(map(check_rel, δ_rel_1))
        @test all(map(check_abs, δ_abs_2)) && all(map(check_rel, δ_rel_2))
        @test all(map(check_abs, δ_abs_3)) && all(map(check_rel, δ_rel_3))
        @test all(map(check_abs, δ_abs_4)) && all(map(check_rel, δ_rel_4))
    end

    # Testing for symm with all four permutations of inputs.
    let α = randn(), A = full(Symmetric(randn(3, 3))), B1 = randn(3, 2), B2 = randn(2, 3)
        diff = [false, false, true, true, true]
        δ_abs_1, δ_rel_1 = discrepancy(BLAS.symm, ('L', 'L', α, A, B1), δ, diff)
        δ_abs_2, δ_rel_2 = discrepancy(BLAS.symm, ('L', 'U', α, A, B1), δ, diff)
        δ_abs_3, δ_rel_3 = discrepancy(BLAS.symm, ('R', 'L', α, A, B2), δ, diff)
        δ_abs_4, δ_rel_4 = discrepancy(BLAS.symm, ('R', 'U', α, A, B2), δ, diff)

        check_abs(x) = all(x .< ϵ_abs)
        check_rel(x) = all(x .< ϵ_rel)
        @test all(map(check_abs, δ_abs_1)) && all(map(check_rel, δ_rel_1))
        @test all(map(check_abs, δ_abs_2)) && all(map(check_rel, δ_rel_2))
        @test all(map(check_abs, δ_abs_3)) && all(map(check_rel, δ_rel_3))
        @test all(map(check_abs, δ_abs_4)) && all(map(check_rel, δ_rel_4))
    end

    # Testing for symv with both permutations.
    let α = randn(), A = full(Symmetric(randn(3, 3))), B = randn(3)
        diff = [false, true, true, true]
        δ_abs_1, δ_rel_1 = discrepancy(BLAS.symv, ('L', α, A, B), δ, diff)
        δ_abs_2, δ_rel_2 = discrepancy(BLAS.symv, ('U', α, A, B), δ, diff)

        check_abs(x) = all(x .< ϵ_abs)
        check_rel(x) = all(x .< ϵ_rel)
        @test all(map(check_abs, δ_abs_1)) && all(map(check_rel, δ_rel_1))
        @test all(map(check_abs, δ_abs_2)) && all(map(check_rel, δ_rel_2))
    end

    # Testing for trmm.
    let α = randn(), A = randn(3, 3), B = randn(3, 3)
        check_abs(x) = all(x .< ϵ_abs)
        check_rel(x) = all(x .< ϵ_rel)
        diff = [false, false, false, false, true, true, true]
        for side in ['L', 'R'], ul in ['L', 'U'], tA in ['N', 'T'], dA in ['N']
            δ_abs, δ_rel = discrepancy(BLAS.trmm, (side, ul, tA, dA, α, A, B), δ, diff)
            @test all(map(check_abs, δ_abs)) && all(map(check_rel, δ_rel))
        end
    end

    # Testing for trmv.
    let A = randn(3, 3), B = randn(3)
        check_abs(x) = all(x .< ϵ_abs)
        check_rel(x) = all(x .< ϵ_rel)
        diff = [false, false, false, true, true]
        for ul in ['L', 'U'], tA in ['N', 'T'], dA in ['N']
            δ_abs, δ_rel = discrepancy(BLAS.trmv, (ul, tA, dA, A, B), δ, diff)
            @test all(map(check_abs, δ_abs)) && all(map(check_rel, δ_rel))
        end
    end

    # Testing for trsm.
    let α = randn(), A = randn(3, 3), B = randn(3, 3)
        check_abs(x) = all(x .< ϵ_abs)
        check_rel(x) = all(x .< ϵ_rel)
        diff = [false, false, false, false, true, true, true]
        for side in ['L', 'R'], ul in ['L', 'U'], tA in ['N', 'T'], dA in ['N']
            δ_abs, δ_rel = discrepancy(BLAS.trsm, (side, ul, tA, dA, α, A, B), δ, diff)
            @test all(map(check_abs, δ_abs)) && all(map(check_rel, δ_rel))
        end
    end

    # Testing for trsv.
    let A = randn(3, 3), B = randn(3)
        check_abs(x) = all(x .< ϵ_abs)
        check_rel(x) = all(x .< ϵ_rel)
        diff = [false, false, false, true, true]
        for ul in ['L', 'U'], tA in ['N', 'T'], dA in ['N']
            δ_abs, δ_rel = discrepancy(BLAS.trsv, (ul, tA, dA, A, B), 1e-8, diff)
            @test all(map(check_abs, δ_abs)) && all(map(check_rel, δ_rel))
        end
    end

end

println("passing.")
