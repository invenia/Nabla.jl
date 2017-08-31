import Nabla: ∇, compute_Dv, approximate_Dv, compute_Dv_update
@testset "finite_differencing" begin

    let
        # Define a dummy test function.
        foo(x::∇Scalar) = 5x
        foo(x::∇Scalar, y::∇Scalar) = 5x + 6y
        foo(x::Vector{<:∇Scalar}, y::Vector{<:∇Scalar}) = 10x + 11y
        foo(x::Matrix{<:∇Scalar}, y::Matrix{<:∇Scalar}) = 15x + 16y

        # Create sensitivity intercepts.
        @explicit_intercepts foo Tuple{∇Scalar}
        @explicit_intercepts foo Tuple{∇Scalar, ∇Scalar}
        @explicit_intercepts foo Tuple{Vector{<:∇Scalar}, Vector{<:∇Scalar}}
        @explicit_intercepts foo Tuple{Matrix{<:∇Scalar}, Matrix{<:∇Scalar}}

        # Define sensitivity implementations.
        const _Vec = Vector{<:∇Scalar}
        const _Mat = Matrix{<:∇Scalar}
        Nabla.∇(::typeof(foo), ::Type{Arg{1}}, p, z, z̄, x::∇Scalar) = 5z̄
        Nabla.∇(::typeof(foo), ::Type{Arg{1}}, p, z, z̄, x::∇Scalar, y::∇Scalar) = 5z̄
        Nabla.∇(::typeof(foo), ::Type{Arg{1}}, p, z, z̄, x::_Vec, y::_Vec) = 10z̄
        Nabla.∇(::typeof(foo), ::Type{Arg{1}}, p, z, z̄, x::_Mat, y::_Mat) = 15z̄
        Nabla.∇(::typeof(foo), ::Type{Arg{2}}, p, z, z̄, x::∇Scalar, y::∇Scalar) = 6z̄
        Nabla.∇(::typeof(foo), ::Type{Arg{2}}, p, z, z̄, x::_Vec, y::_Vec) = 11z̄
        Nabla.∇(::typeof(foo), ::Type{Arg{2}}, p, z, z̄, x::_Mat, y::_Mat) = 16z̄

        # Check that Nabla and finite differencing yield the correct results for scalars.
        @test compute_Dv(foo, 1.0, 1.0, 1.0) ≈ 5
        @test compute_Dv_update(foo, 1.0, 1.0, 1.0) ≈ 5
        @test approximate_Dv(foo, 1.0, 1.0, 1.0) ≈ 5
        @test compute_Dv(foo, 1.0, (1.0, 1.0), (1.0, 1.0)) ≈ 5 + 6
        @test compute_Dv_update(foo, 1.0, (1.0, 1.0), (1.0, 1.0)) ≈ 5 + 6
        @test approximate_Dv(foo, 1.0, (1.0, 1.0), (1.0, 1.0)) ≈ 5 + 6
        @test compute_Dv(foo, 5.0, (1.0, 1.0), (1.0, 1.0)) ≈ (5 + 6) * 5.0
        @test compute_Dv_update(foo, 5.0, (1.0, 1.0), (1.0, 1.0)) ≈ (5 + 6) * 5.0
        @test approximate_Dv(foo, 5.0, (1.0, 1.0), (1.0, 1.0)) ≈ (5 + 6) * 5.0
        @test compute_Dv(foo, 1.0, (10.0, 5.0), (1.0, 1.0)) ≈ 5 + 6
        @test compute_Dv_update(foo, 1.0, (10.0, 5.0), (1.0, 1.0)) ≈ 5 + 6
        @test approximate_Dv(foo, 1.0, (10.0, 5.0), (1.0, 1.0)) ≈ 5 + 6
        @test compute_Dv(foo, 1.0, (1.0, 1.0), (7.5, 6.3)) ≈ 5 * 7.5 + 6 * 6.3
        @test compute_Dv_update(foo, 1.0, (1.0, 1.0), (7.5, 6.3)) ≈ 5 * 7.5 + 6 * 6.3
        @test approximate_Dv(foo, 1.0, (1.0, 1.0), (7.5, 6.3)) ≈ 5 * 7.5 + 6 * 6.3
        @test compute_Dv(foo, 5.0, (10.0, 5.0), (7.5, 6.3)) ≈ (5 * 7.5 + 6 * 6.3) * 5
        @test compute_Dv_update(foo, 5.0, (10.0, 5.0), (7.5, 6.3)) ≈ (5 * 7.5 + 6 * 6.3) * 5
        @test approximate_Dv(foo, 5.0, (10.0, 5.0), (7.5, 6.3)) ≈ (5 * 7.5 + 6 * 6.3) * 5

        # Check that finite differencing yields approximately correct results for vectors.
        N = 3
        z̄, x, y, vx, vy = ones.((N, N, N, N, N))
        @test compute_Dv(foo, z̄, (x, y), (vx, vy)) ≈
            sum(z̄ .* (UniformScaling(10) * vx)) + sum(z̄ .* (UniformScaling(11) * vy))
        @test compute_Dv_update(foo, z̄, (x, y), (vx, vy)) ≈
            sum(z̄ .* (UniformScaling(10) * vx)) + sum(z̄ .* (UniformScaling(11) * vy))
        @test approximate_Dv(foo, z̄, (x, y), (vx, vy)) ≈
            sum(z̄ .* (UniformScaling(10) * vx)) + sum(z̄ .* (UniformScaling(11) * vy))
        z̄ = randn(N)
        @test compute_Dv(foo, z̄, (x, y), (vx, vy)) ≈
            sum(z̄ .* (UniformScaling(10) * vx)) + sum(z̄ .* (UniformScaling(11) * vy))
        @test compute_Dv_update(foo, z̄, (x, y), (vx, vy)) ≈
            sum(z̄ .* (UniformScaling(10) * vx)) + sum(z̄ .* (UniformScaling(11) * vy))
        @test approximate_Dv(foo, z̄, (x, y), (vx, vy)) ≈
            sum(z̄ .* (UniformScaling(10) * vx)) + sum(z̄ .* (UniformScaling(11) * vy))
        v = randn(N)
        @test compute_Dv(foo, z̄, (x, y), (vx, vy)) ≈
            sum(z̄ .* (UniformScaling(10) * vx)) + sum(z̄ .* (UniformScaling(11) * vy))
        @test compute_Dv_update(foo, z̄, (x, y), (vx, vy)) ≈
            sum(z̄ .* (UniformScaling(10) * vx)) + sum(z̄ .* (UniformScaling(11) * vy))
        @test approximate_Dv(foo, z̄, (x, y), (vx, vy)) ≈
            sum(z̄ .* (UniformScaling(10) * vx)) + sum(z̄ .* (UniformScaling(11) * vy))

        # Check that finite differencing yields approximately correct results for matrices.
        M, N = 3, 2
        Z̄, X, Y, VX, VY = ones.((M, M, M, M, M), (N, N, N, N, N))
        @test approximate_Dv(foo, Z̄, (X, Y), (VX, VY)) ≈
            compute_Dv(foo, Z̄, (X, Y), (VX, VY))
        @test compute_Dv(foo, Z̄, (X, Y), (VX, VY)) ≈
            compute_Dv_update(foo, Z̄, (X, Y), (VX, VY))
        Ȳ = randn(M, N)
        @test approximate_Dv(foo, Z̄, (X, Y), (VX, VY)) ≈
            compute_Dv(foo, Z̄, (X, Y), (VX, VY))
        @test compute_Dv(foo, Z̄, (X, Y), (VX, VY)) ≈
            compute_Dv_update(foo, Z̄, (X, Y), (VX, VY))
    end
end

function print_tol_err(f, ȳ, x::T, v::T, err_abs::∇Scalar, err_rel::∇Scalar) where T<:∇ArrayOrScalar
    println("Large error found in sensitivity for function $f at input")
    println(x)
    println("in direction")
    println(v)
    println("err_abs = $err_abs, err_rel = $err_rel")
    throw(error("Large error found in sensitivity."))
end
