@testset "sensitivities/linalg/generic" begin

    let ϵ_abs = 1e-6, c_rel = 1e6, N = 5, rng = MersenneTwister(123456)

        # Generate random test quantities for specific types.
        ∇Arrays = Union{Type{∇RealArray}, Type{ArrayOr∇Real}}
        trandn(rng::AbstractRNG, ::∇Arrays) = randn(rng, N, N)
        trandn(rng::AbstractRNG, ::Type{∇Real}) = randn(rng)
        trand(rng::AbstractRNG, ::∇Arrays) = rand(rng, N, N)
        trand(rng::AbstractRNG, ::Type{∇Real}) = rand(rng)

        # Test unary linalg sensitivities.
        for (f, T_In, T_Out, X̄, bounds) in Nabla.unary_linalg_optimisations
            Z = trand(rng, T_In) .* (bounds[2] - bounds[1]) + bounds[1]
            X = Z'Z + 1e-6 * one(Z)
            Ȳ, V = eval(f)(X), 1e-3 * trandn(rng, T_In)
            @test check_errs(eval(f), Ȳ, X, V'V + 1e-6 * one(V), ϵ_abs, c_rel)
        end

        # Test binary linalg sensitivities.
        for (f, T_A, T_B, T_Y, Ā, B̄) in Nabla.binary_linalg_optimisations
            A, B, VA, VB = trandn.(rng, (T_A, T_B, T_A, T_B))
            @test check_errs(eval(f), eval(f)(A, B), (A, B), 1e-6 .* (VA, VB), ϵ_abs, c_rel)
        end
    end
end
