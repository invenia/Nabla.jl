@testset "sensitivities/linalg/generic" begin

    let ϵ_abs = 1e-6, c_rel = 1e6, N = 5, rng = MersenneTwister(123456)

        # Test unary linalg sensitivities.
        for (f, T_In, T_Out, X̄, bounds) in Nabla.unary_linalg_optimisations
            Z = rand(rng, N, N) .* (bounds[2] - bounds[1]) + bounds[1]
            X = Z'Z + 1e-6I
            Ȳ, V_ = eval(f)(X), 1e-3 * randn(rng, size(X))
            @test check_errs(eval(f), Ȳ, X, V_.'V_ + 1e-6I, ϵ_abs, c_rel)
        end

        # Test binary linalg optimisations.
        for (f, T_A, T_B, T_Y, Ā, B̄, errs) in Nabla.binary_linalg_optimisations
            A, B, VA, VB = randn.(rng, (N, N, N, N), (N, N, N, N))
            @test check_errs(eval(f), eval(f)(A, B), (A, B), 1e-6 .* (VA, VB), ϵ_abs, c_rel)
        end
    end
end
