print("sensitivities/blas.jl... ")

let ϵ_abs = 1e-5, ϵ_rel = 1e-4, δ = 1e-6

    # Testing for dot.
    let x = randn(10), y = randn(6)
        δ_abs, δ_rel = @eval discrepancy(dot, ($(x[1:5]), $(y[1:5])), $δ)
        @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel) &&
              all(δ_abs[2] .< ϵ_abs) && all(δ_rel[2] .< ϵ_rel)

        diff = [false, true, false, true, false]
        δ_abs, δ_rel = @eval discrepancy(dot, (5, $x, 2, $y, 1), $δ, $diff)
        @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel) &&
              all(δ_abs[2] .< ϵ_abs) && all(δ_rel[2] .< ϵ_rel)
    end

    # Testing for nrm2.
    let x = randn(10)
        δ_abs, δ_rel = @eval discrepancy(BLAS.nrm2, ($x,), $δ)
        @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)

        δ_abs, δ_rel = @eval discrepancy(BLAS.nrm2, (5, $x, 2), $δ, [false, true, false])
        @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
    end

    # Testing for asum.
    let x = randn(10)
        δ_abs, δ_rel = @eval discrepancy(BLAS.asum, ($x,), $δ)
        @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)

        δ_abs, δ_rel = @eval discrepancy(BLAS.asum, (5, $x, 2), $δ, [false, true, false])
        @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
    end

    # # Testing for scal.
    # let x = randn(10)
    #     δ_abs, δ_rel = @eval discrepancy(BLAS.scal, (5, 2.5, $x, 2), $δ, [false, true, true, false])
    #     @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
    # end

end

println("passing.")
