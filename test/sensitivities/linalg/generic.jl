@testset "sensitivities/linalg/generic" begin

    let ϵ_abs = 1e-18, ϵ_rel = 1e-18, N = 25

        # Test unary linalg sensitivities.
        for (f, T_In, T_Out, X̄, bounds) in Nabla.unary_linalg_optimisations

            # Test allocating sensitivities.
            Z = rand(Uniform(bounds[1], bounds[2]), N, N)
            X = Z'Z + UniformScaling(1e-3)
            Ȳ, V_ = eval(f)(X), 1e-3 * randn(size(X))
            δ_abs, δ_rel = check_Dv(eval(f), Ȳ, X, V_.'V_)

            # Check that sensitivities w.r.t. the input are correct.
            (δ_abs > ϵ_abs || δ_rel > ϵ_rel || isnan(δ_abs) || isnan(δ_rel)) &&
                Nabla.print_tol_err(eval(f), Ȳ, X, V, δ_abs, δ_rel)
            @test δ_abs < ϵ_abs && δ_rel < ϵ_rel
        end

        # Test binary linalg optimisations.
        for (f, T_A, T_B, T_Y, Ā, B̄, errs) in Nabla.binary_linalg_optimisations

            # Construct λ-functions for each argument.
            f, A, B = eval(f), randn(N, N), randn(N, N)
            λA, λB = A->f(A, B), B->f(A, B)
            Ȳ, V = f(A, B), 1e-6 * randn(N, N)

            # Compute errors w.r.t. first argument.
            δ_abs_A, δ_rel_A = check_Dv(λA, Ȳ, A, V)
            (δ_abs_A > ϵ_abs || δ_rel_A > ϵ_rel || isnan(δ_abs_A) || isnan(δ_rel_A)) &&
                Nabla.print_tol_err(λA, Ȳ, A, V, δ_abs_A, δ_rel_A)
            @test δ_abs_A < ϵ_abs && δ_rel_A < ϵ_rel

            # Compute errors w.r.t. second argument.
            δ_abs_B, δ_rel_B = check_Dv(λB, Ȳ, B, V)
            (δ_abs_B > ϵ_abs || δ_rel_B > ϵ_rel || isnan(δ_abs_B) || isnan(δ_rel_B)) &&
                Nabla.print_tol_err(λB, Ȳ, B, V, δ_abs_B, δ_rel_B)
            @test δ_abs_B < ϵ_abs && δ_rel_B < ϵ_rel
        end
    end
end
