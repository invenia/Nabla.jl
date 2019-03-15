@testset "Generic" begin

    let N = 5, rng = MersenneTwister(123456)

        # Generate random test quantities for specific types.
        ∇Arrays = Union{Type{∇Array}, Type{∇ArrayOrScalar}, Type{<:Transpose}, Type{<:Adjoint}}
        trandn(rng::AbstractRNG, ::∇Arrays) = randn(rng, N, N)
        trandn2(rng::AbstractRNG, ::∇Arrays) = randn(rng, N^2, N^2)
        trandn(rng::AbstractRNG, ::Type{∇Scalar}) = randn(rng)
        trand(rng::AbstractRNG, ::∇Arrays) = rand(rng, N, N)
        trand(rng::AbstractRNG, ::Type{∇Scalar}) = rand(rng)

        for _ in 1:5
            # Test unary linalg sensitivities.
            for (f, T_In, T_Out, X̄, bounds) in Nabla.unary_linalg_optimisations
                Z = trand(rng, T_In) .* (bounds[2] .- bounds[1]) .+ bounds[1]
                X = Z'Z + 1e-6 * one(Z)
                Ȳ, V = eval(f)(X), trandn(rng, T_In)
                @test check_errs(eval(f), Ȳ, X, 1e-1 .* V)
            end

            # Test binary linalg sensitivities.
            for (f, T_A, T_B, T_Y, Ā, B̄) in Nabla.binary_linalg_optimisations
                A, B, VA, VB = trandn.(Ref(rng), (T_A, T_B, T_A, T_B))
                @test check_errs(eval(f), eval(f)(A, B), (A, B), (VA, VB))
            end
            A, B, VA, VB = trandn.(Ref(rng), (∇Array, ∇Array, ∇Array, ∇Array))
            @test check_errs(kron, kron(A, B), (A, B), (VA, VB))
        end

        for _ in 1:5
            A, VA, tI = randn(rng, 5, 5), randn(rng, 5, 5), 0.65 * I
            @test check_errs(X->X + tI, VA, A, 1e-1 * randn(rng, 5, 5))
            @test check_errs(X->tI + X, VA, A, 1e-1 * randn(rng, 5, 5))
        end

    end

    # dot
    let rng = MersenneTwister(123456)
        for _ in 1:10
            x, y, vx, vy = randn.(Ref(rng), [5, 5, 5, 5])
            @test check_errs(LinearAlgebra.dot, LinearAlgebra.dot(x, y), (x, y), (vx, vy))
        end
    end

    @testset "copy" begin
        rng = MersenneTwister(12345)

        # Scalars (no-op)
        x = randn(rng)
        y = randn(rng)
        @test check_errs(copy, x, x, y)
        x_ = Leaf(Tape(), x)
        c = copy(x_)
        @test c isa Branch{Float64}
        @test getfield(c, :f) === Base.copy

        # Unwrapping adjoint/transposes
        X = randn(rng, 6, 6)'
        Y = randn(rng, 6, 6)
        @test check_errs(copy, X, copy(X), Y)
        X_ = Leaf(Tape(), X)
        C = copy(X_)
        @test C isa Branch{Matrix{Float64}}
        @test getfield(c, :f) === Base.copy
    end
end
