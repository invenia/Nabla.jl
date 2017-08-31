@testset "sensitivities/linalg/generic" begin

    let ϵ_abs = 1e-6, c_rel = 1e6, N = 5, rng = MersenneTwister(123456)

        # Generate random test quantities for specific types.
        for f in [:rand, :randn]
            @eval Base.$f(rng::AbstractRNG, ::Type{∇Real}, N::Int64) = $f(rng)
            @eval Base.$f(rng::AbstractRNG, ::Type{∇RealArray}, N::Int64) = $f(rng, N, N)
            @eval Base.$f(rng::AbstractRNG, ::Type{ArrayOr∇Real}, N::Int64) = $f(rng, ∇RealArray, N)
        end

        # Get the identity associated to an object.
        id(::Any) = I
        id(::Number) = I.λ

        # Test unary linalg sensitivities.
        for (f, T_In, T_Out, X̄, bounds) in Nabla.unary_linalg_optimisations
            Z = rand(rng, T_In, N) .* (bounds[2] - bounds[1]) + bounds[1]
            X = Z'Z + 1e-6id(Z)
            Ȳ, V = eval(f)(X), 1e-3 * randn(rng, T_In, N)
            @test check_errs(eval(f), Ȳ, X, V'V + 1e-6id(V), ϵ_abs, c_rel)
        end

        # Test binary linalg sensitivities.
        for (f, T_A, T_B, T_Y, Ā, B̄) in Nabla.binary_linalg_optimisations
            A, B, VA, VB = randn.(rng, (T_A, T_B, T_A, T_B), N)
            @test check_errs(eval(f), eval(f)(A, B), (A, B), 1e-6 .* (VA, VB), ϵ_abs, c_rel)
        end
    end
end
