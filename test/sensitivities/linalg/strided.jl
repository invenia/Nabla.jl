@testset "sensitivities/linalg/strided" begin

    let ϵ_abs = 1e-5, ϵ_rel = 1e-5, N = 5

        # Test strided matrix-matrix multiplication sensitivities.
        for (f, tCA, tDA, CA, DA, tCB, tDB, CB, DB) in Nabla.strided_matmul

            # Create input arrays and λ-functions for each input.
            f, A, B = eval(f), randn(N, N), randn(N, N)
            λA, λB = A->f(A, B), B->f(A, B)
            Ȳ, V = f(A, B), 1e-6 * randn(N, N)

            # Test allocating sensitivites on first arg.
            δ_abs_A, δ_rel_A = check_Dv(λA, Ȳ, A, V)
            (δ_abs_A > ϵ_abs || δ_rel_A > ϵ_rel || isnan(δ_abs_A) || isnan(δ_rel_A)) &&
                Nabla.print_tol_err(λA, Ȳ, A, V, δ_abs_A, δ_rel_A)
            @test δ_abs_A < ϵ_abs && δ_rel_A < ϵ_rel

            # Compute errors w.r.t. second argument.
            δ_abs_B, δ_rel_B = check_Dv(λB, Ȳ, B, V)
            (δ_abs_B > ϵ_abs || δ_rel_B > ϵ_rel || isnan(δ_abs_B) || isnan(δ_rel_B)) &&
                Nabla.print_tol_err(λB, Ȳ, B, V, δ_abs_B, δ_rel_B)
            @test δ_abs_B < ϵ_abs && δ_rel_B < ϵ_rel

            # THESE TESTS NEED TO BE REINSTANTIATED AT SOME POINT BEFORE RELEASE.
            # # Test allocating sensitivities.
            # A, B, tape = randn(N, N), randn(N, N), Tape()
            # A_, B_ = Leaf(tape, A), Leaf(tape, B)
            # δ_abs, δ_rel = @eval discrepancy($f, ($A, $B), $δ)
            # @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
            # @test all(δ_abs[2] .< ϵ_abs) && all(δ_rel[2] .< ϵ_rel)

            # # Test non-allocating sensitivities.
            # g(A, B) = @eval $f($f($A, $f($A, $B)), $B)
            # δ_abs, δ_rel = discrepancy(g, (A, B), δ)
            # @test g(A, B) == g(A_, B_).val
            # @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
            # @test all(δ_abs[2] .< ϵ_abs) && all(δ_rel[2] .< ϵ_rel)
        end

        # # Test matrix-vector multiplication sensitivities.
        # for (f, tdA, CA, dA, tCb) in Nabla.strided_matvecmul

        #     # Test allocating sensitivities.
        #     A, b = randn(N, N), randn(N)
        #     δ_abs, δ_rel = @eval discrepancy($f, ($A, $b), $δ)
        #     @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
        #     @test all(δ_abs[2] .< ϵ_abs) && all(δ_rel[2] .< ϵ_rel)

        #     # Test non-allocating sensitivities.
        #     g(A, b) = @eval $f($A, $f($A, $b)) .+ $f($A, $b)
        #     δ_abs, δ_rel = discrepancy(g, (A, b), δ)
        #     @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
        #     @test all(δ_abs[2] .< ϵ_abs) && all(δ_rel[2] .< ϵ_rel)
        # end

        # # Test solving sensitivities.
        # for (f, C, tA, tB, arg1, arg2) in Nabla.strided_ldiv

        #     # Test allocating sensitivities.
        #     A, B = randn(5, 5), randn(5, 5)
        #     δ_abs, δ_rel = @eval discrepancy($f, ($A, $B), $δ)
        #     @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
        #     @test all(δ_abs[2] .< ϵ_abs) && all(δ_rel[2] .< ϵ_rel)
        # end
    end
end
