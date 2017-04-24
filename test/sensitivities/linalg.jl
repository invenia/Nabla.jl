print("sensitivities/linalg.jl... ")

let ϵ_abs = 1e-3, ϵ_rel = 1e-2, δ = 1e-6, N = 5

    # Test matrix-matrix multiplication sensitivities.
    for (f, new_Ā, update_Ā, new_B̄, update_B̄) in AutoGrad2.strided_matmul

        # Test allocating sensitivities.
        A, B = randn(N, N), randn(N, N)
        δ_abs, δ_rel = @eval discrepancy($f, ($A, $B), $δ)
        @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
        @test all(δ_abs[2] .< ϵ_abs) && all(δ_rel[2] .< ϵ_rel)

        # Test non-allocating sensitivities.
        g(A, B) = @eval $f($f($A, $f($A, $B)), $B)
        δ_abs, δ_rel = discrepancy(g, (A, B), δ)
        @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
        @test all(δ_abs[2] .< ϵ_abs) && all(δ_rel[2] .< ϵ_rel)
    end

    # Test matrix-vector multiplication sensitivities.
    for (f, tdA, CA, dA, tCb) in AutoGrad2.strided_matvecmul

        # Test allocating sensitivities.
        A, b = randn(N, N), randn(N)
        δ_abs, δ_rel = @eval discrepancy($f, ($A, $b), $δ)
        @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
        @test all(δ_abs[2] .< ϵ_abs) && all(δ_rel[2] .< ϵ_rel)

        # Test non-allocating sensitivities.
        g(A, b) = @eval $f($A, $f($A, $b)) .+ $f($A, $b)
        δ_abs, δ_rel = discrepancy(g, (A, b), δ)
        @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
        @test all(δ_abs[2] .< ϵ_abs) && all(δ_rel[2] .< ϵ_rel)
    end

    # Test generic multiplication sensitivities.
    for (f, new_Ā, update_Ā, new_B̄, update_B̄) in AutoGrad2.strided_matmul

        # Test allocating sensitivities.
        A, B = sparse(randn(N, N)), sparse(randn(N, N))
        δ_abs, δ_rel = @eval discrepancy($f, ($A, $B), $δ)
        @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
        @test all(δ_abs[2] .< ϵ_abs) && all(δ_rel[2] .< ϵ_rel)

        # Test non-allocating sensitivities.
        g(A, B) = @eval $f($f($A, $f($A, $B)), $B)
        δ_abs, δ_rel = discrepancy(g, (A, B), δ)
        @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
        @test all(δ_abs[2] .< ϵ_abs) && all(δ_rel[2] .< ϵ_rel)
    end

    # Test solving sensitivities.
    for (f, C, tA, tB, arg1, arg2) in AutoGrad2.strided_ldiv

        # Test allocating sensitivities.
        A, B = randn(5, 5), randn(5, 5)
        δ_abs, δ_rel = @eval discrepancy($f, ($A, $B), $δ)
        @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
        @test all(δ_abs[2] .< ϵ_abs) && all(δ_rel[2] .< ϵ_rel)
    end
end

println("passing.")
