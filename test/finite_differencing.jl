@testset "finite_differencing" begin
    let
        # Define a dummy test function.
        foo(x::∇Real) = 5x
        foo(x::Vector{<:∇Real}) = 10x
        foo(x::Matrix{<:∇Real}) = 15x

        # Create sensitivity intercepts.
        @explicit_intercepts foo Tuple{∇Real}
        @explicit_intercepts foo Tuple{Vector{<:∇Real}}
        @explicit_intercepts foo Tuple{Matrix{<:∇Real}}

        # Define sensitivity implementations.
        Nabla.∇(::typeof(foo), ::Type{Arg{1}}, p, y, ȳ, x::∇Real) = 5ȳ
        Nabla.∇(::typeof(foo), ::Type{Arg{1}}, p, y, ȳ, x::Vector{<:∇Real}) = 10ȳ
        Nabla.∇(::typeof(foo), ::Type{Arg{1}}, p, y, ȳ, x::Matrix{<:∇Real}) = 15ȳ

        # Check that Nabla and finite differencing yield the correct results for scalars.
        @test Nabla.compute_Dv(foo, 1.0, 1.0, 1.0) == 5
        @test Nabla.approximate_Dv(foo, 1.0, 1.0, 1.0) == 5
        @test Nabla.compute_Dv(foo, 5.0, 1.0, 1.0) == 5 * 5.0
        @test Nabla.approximate_Dv(foo, 5.0, 1.0, 1.0) == 5 * 5.0
        @test Nabla.compute_Dv(foo, 1.0, 10.0, 1.0) == 5
        @test Nabla.approximate_Dv(foo, 1.0, 10.0, 1.0) == 5
        @test Nabla.compute_Dv(foo, 1.0, 1.0, 7.5) == 5 * 7.5
        @test Nabla.approximate_Dv(foo, 1.0, 1.0, 7.5) == 5 * 7.5
        @test Nabla.compute_Dv(foo, 5.0, 10.0, 7.5) == 5.0 * 7.5 * 5
        @test Nabla.approximate_Dv(foo, 5.0, 10.0, 7.5) == 5.0 * 7.5 * 5

        # Check that finite differencing yields approximately correct results for vectors.
        N = 3
        ȳ, x, v = ones.((N, N, N))
        @test Nabla.compute_Dv(foo, ȳ, x, v) == sum(ȳ .* (UniformScaling(10) * v))
        @test Nabla.approximate_Dv(foo, ȳ, x, v) == sum(ȳ .* (UniformScaling(10) * v))
        ȳ = randn(N)
        @test Nabla.compute_Dv(foo, ȳ, x, v) == sum(ȳ .* (UniformScaling(10) * v))
        @test Nabla.approximate_Dv(foo, ȳ, x, v) == sum(ȳ .* (UniformScaling(10) * v))
        v = randn(N)
        @test Nabla.compute_Dv(foo, ȳ, x, v) ≈ sum(ȳ .* (UniformScaling(10) * v))
        @test Nabla.approximate_Dv(foo, ȳ, x, v) ≈ sum(ȳ .* (UniformScaling(10) * v))

        # Check that finite differencing yields approximately correct results for matrices.
        M, N = 3, 2
        Ȳ, X, V = ones.((M, M, M), (N, N, N))
        @test Nabla.approximate_Dv(foo, Ȳ, X, V) == Nabla.compute_Dv(foo, Ȳ, X, V)
        Ȳ = randn(M, N)
        @test Nabla.approximate_Dv(foo, Ȳ, X, V) == Nabla.compute_Dv(foo, Ȳ, X, V)
    end

    function print_tol_err(f, ȳ, x::T, v::T, err_abs::∇Real, err_rel::∇Real) where T<:ArrayOr∇Real
        println("Large error found in sensitivity for function $f at input")
        println(x)
        println("in direction")
        println(v)
        println("err_abs = $err_abs, err_rel = $err_rel")
        throw(error("Large error found in sensitivity."))
    end
end
