@testset "Generic" begin

    let N = 5, rng = MersenneTwister(123456)

        # Generate random test quantities for specific types.
        ∇Arrays = Union{Type{∇Array}, Type{∇ArrayOrScalar}}
        trandn(rng::AbstractRNG, ::∇Arrays) = randn(rng, N, N)
        trandn(rng::AbstractRNG, ::Type{∇Scalar}) = randn(rng)
        trand(rng::AbstractRNG, ::∇Arrays) = rand(rng, N, N)
        trand(rng::AbstractRNG, ::Type{∇Scalar}) = rand(rng)

        # Test unary linalg sensitivities.
        for (f, T_In, T_Out, X̄, bounds) in Nabla.unary_linalg_optimisations
            Z = trand(rng, T_In) .* (bounds[2] - bounds[1]) + bounds[1]
            X = Z'Z + 1e-6 * one(Z)
            Ȳ, V = eval(f)(X), trandn(rng, T_In)
            V'V + 1e-6 * one(V)
            @test check_errs(eval(f), Ȳ, X, 1e-2 .* V)
        end

        # Test binary linalg sensitivities.
        for (f, T_A, T_B, T_Y, Ā, B̄) in Nabla.binary_linalg_optimisations
            A, B, VA, VB = trandn.(rng, (T_A, T_B, T_A, T_B))
            @test check_errs(eval(f), eval(f)(A, B), (A, B), (VA, VB))
        end
    end
end
