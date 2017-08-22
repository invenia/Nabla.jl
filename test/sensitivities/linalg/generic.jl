@testset "sensitivities/linalg/generic" begin

    let ϵ_abs = 1e-5, ϵ_rel = 1e-6, N = 5, rng = MersenneTwister(123456)

        # Test unary linalg sensitivities.
        for (f, T_In, T_Out, X̄, bounds) in Nabla.unary_linalg_optimisations

            # Test allocating sensitivities.
            Z = rand(rng, N, N) .* (bounds[2] - bounds[1]) + bounds[1]
            X = Z'Z + UniformScaling(1e-6)
            Ȳ, V_ = eval(f)(X), 1e-3 * randn(rng, size(X))
            δ_abs, δ_rel = check_Dv(eval(f), Ȳ, X, V_.'V_ + UniformScaling(1e-6))

            # Check that sensitivities w.r.t. the input are correct.
            (δ_abs > ϵ_abs || δ_rel > ϵ_rel || isnan(δ_abs) || isnan(δ_rel)) &&
                print_tol_err(eval(f), Ȳ, X, V_, δ_abs, δ_rel)
            @test δ_abs < ϵ_abs && δ_rel < ϵ_rel
        end

        # Test binary linalg optimisations.
        for (f, T_A, T_B, T_Y, Ā, B̄, errs) in Nabla.binary_linalg_optimisations

            # Construct λ-functions for each argument.
            A, B, VA, VB = randn.(rng, (N, N, N, N), (N, N, N, N))
            δ_abs, δ_rel = check_Dv(eval(f), eval(f)(A, B), (A, B), 1e-6 .* (VA, VB))
            (δ_abs > ϵ_abs || δ_rel > ϵ_rel || isnan(δ_abs) || isnan(δ_rel)) &&
                print_tol_err(eval(f), eval(f)(A, B), A, VA, δ_abs, δ_rel)
            @test δ_abs < ϵ_abs && δ_rel < ϵ_rel
        end
    end
end
