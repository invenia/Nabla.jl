@testset "Generic" begin
    N = 5

    # Generate random test quantities for specific types.
    ∇Arrays = Union{Type{∇Array}, Type{∇ArrayOrScalar}}

    trandn(rng::AbstractRNG, ::∇Arrays) = randn(rng, N, N)
    trandn(rng::AbstractRNG, ::Type{∇Scalar}) = randn(rng)
    trandn(rng::AbstractRNG, ::Type{<:Transpose}) = Transpose(randn(rng, N, N))
    trandn(rng::AbstractRNG, ::Type{<:Adjoint}) = Adjoint(randn(rng, N, N))

    trandn2(rng::AbstractRNG, ::∇Arrays) = randn(rng, N^2, N^2)
    trandn2(rng::AbstractRNG, ::Type{<:Transpose}) = Transpose(randn(rng, N^2, N^2))
    trandn2(rng::AbstractRNG, ::Type{<:Adjoint}) = Adjoint(randn(rng, N^2, N^2))

    trand(rng::AbstractRNG, ::∇Arrays) = rand(rng, N, N)
    trand(rng::AbstractRNG, ::Type{∇Scalar}) = rand(rng)
    trand(rng::AbstractRNG, ::Type{<:Transpose}) = Transpose(rand(rng, N, N))
    trand(rng::AbstractRNG, ::Type{<:Adjoint}) = Adjoint(rand(rng, N, N))

    @testset "Unary sensitivities" begin
        _ϵ, lb, ub = 3e-2, -3.0, 3.0
        unary_linalg_optimisations = [
            (-,          ∇Array,  (lb, ub)),
            (tr,         ∇Array,  (lb, ub)),
            (inv,        ∇Array,  (lb, ub)),
            (det,        ∇Array,  (_ϵ, ub)),
            (logdet,     ∇Array,  (_ϵ, ub)),
            (transpose,  ∇Array,  (lb, ub)),
            (adjoint,    ∇Scalar, (_ϵ, ub)),
            (adjoint,    ∇Array,  (lb, ub)),
            (norm,       ∇Array,  (lb, ub)),
            (norm,       ∇Scalar, (lb, ub)),
        ]

        rng = MersenneTwister(123)
        @testset "$f" for (f, T_In, bounds) in unary_linalg_optimisations
            for _ in 1:5
                Z = trand(rng, T_In) .* (bounds[2] .- bounds[1]) .+ bounds[1]
                X = Z'Z + 1e-6 * one(Z)
                Ȳ, V = f(X), trandn(rng, T_In)
                @test check_errs(f, Ȳ, X, 1e-1 .* V)
            end
        end
    end

    @testset "Binary sensitivities" begin
        rng = MersenneTwister(2)
        binary_linalg_optimisations = [
            (*, ∇Array, ∇Array,),
            (/, ∇Array, ∇Array,),
            (\, ∇Array, ∇Array,),
            (norm, ∇Array, ∇Scalar,),
            (norm, ∇Scalar, ∇Scalar,),
        ]
        @testset "$f" for (f, T_A, T_B) in binary_linalg_optimisations
            for _ in 1:5
                A, B, VA, VB = trandn.(Ref(rng), (T_A, T_B, T_A, T_B))
                @test check_errs(f, eval(f)(A, B), (A, B), (VA, VB))
            end
        end
    end

    @testset "kron" begin
        rng = MersenneTwister(3)
        for _ in 1:5
            A, B, VA, VB = trandn.(Ref(rng), (∇Array, ∇Array, ∇Array, ∇Array))
            @test check_errs(kron, kron(A, B), (A, B), (VA, VB))
        end
    end

    @testset "I" begin
        rng = MersenneTwister(4)
        for _ in 1:5
            A, VA, tI = randn(rng, 5, 5), randn(rng, 5, 5), 0.65 * I
            @test check_errs(X->X + tI, VA, A, 1e-1 * randn(rng, 5, 5))
            @test check_errs(X->tI + X, VA, A, 1e-1 * randn(rng, 5, 5))
        end
    end

    @testset "dot" begin
        rng = MersenneTwister(123456)
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

    @testset "exp" begin
        rng = MersenneTwister(12345)
        n = 10
        symm!(X) = (X .= (X .+ X') ./ 2; X)
        X = symm!(randn(rng, n, n))
        VX = symm!(randn(rng, n, n))
        @test check_errs(exp, randn(rng, n, n), X, VX)
        A = randn(rng, n, n)
        VA = randn(rng, n, n)
        @test_throws ArgumentError check_errs(exp, randn(rng, n, n), A, VA)
    end
end
