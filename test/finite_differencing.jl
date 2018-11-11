import Nabla: ∇, compute_Dv, approximate_Dv, compute_Dv_update, is_atom, ∇MaybeTagged, ∇Ctx, Arg
using Cassette: istagged
using LinearAlgebra

# Define a dummy test function.
fd_foo(x::∇Scalar) = 5x
fd_foo(x::∇Scalar, y::∇Scalar) = 5x + 6y
fd_foo(x::Vector{<:∇Scalar}, y::Vector{<:∇Scalar}) = 10x + 11y
fd_foo(x::Matrix{<:∇Scalar}, y::Matrix{<:∇Scalar}) = 15x + 16y

is_atom(ctx::∇Ctx, ::typeof(fd_foo), x::∇MaybeTagged{<:∇Scalar}) = istagged(x, ctx)
function is_atom(
    ctx::∇Ctx,
    ::typeof(fd_foo),
    x::∇MaybeTagged{<:∇Scalar},
    y::∇MaybeTagged{<:∇Scalar},
)
    return istagged(x, ctx) || istagged(y, ctx)
end
function is_atom(
    ctx::∇Ctx,
    ::typeof(fd_foo),
    x::∇MaybeTagged{<:Vector{<:∇Scalar}},
    y::∇MaybeTagged{<:Vector{<:∇Scalar}},
)
    return istagged(x, ctx) || istagged(y, ctx)
end
function is_atom(
    ctx::∇Ctx,
    ::typeof(fd_foo),
    x::∇MaybeTagged{<:Matrix{<:∇Scalar}},
    y::∇MaybeTagged{<:Matrix{<:∇Scalar}},
)
    return istagged(x, ctx) || istagged(y, ctx)
end

# Define sensitivity implementations. These are _intentionally_ incorrect.
const _Vec = Vector{<:∇Scalar}
const _Mat = Matrix{<:∇Scalar}
const fd_a = 3.2
const fd_b = 2.5
∇(::typeof(fd_foo), ::Type{Arg{1}}, p, z, z̄, x::∇Scalar) = fd_a * z̄
∇(::typeof(fd_foo), ::Type{Arg{1}}, p, z, z̄, x::∇Scalar, y::∇Scalar) = fd_a * z̄
∇(::typeof(fd_foo), ::Type{Arg{1}}, p, z, z̄, x::_Vec, y::_Vec) = fd_a * z̄
∇(::typeof(fd_foo), ::Type{Arg{1}}, p, z, z̄, x::_Mat, y::_Mat) = fd_a * z̄
∇(::typeof(fd_foo), ::Type{Arg{2}}, p, z, z̄, x::∇Scalar, y::∇Scalar) = fd_b * z̄
∇(::typeof(fd_foo), ::Type{Arg{2}}, p, z, z̄, x::_Vec, y::_Vec) = fd_b * z̄
∇(::typeof(fd_foo), ::Type{Arg{2}}, p, z, z̄, x::_Mat, y::_Mat) = fd_b * z̄

@testset "Finite-difference estimates of sensitivities" begin

    # Check that Nabla and finite differencing yield the correct results for scalars.
    @test compute_Dv(fd_foo, 1.0, 1.0, 1.0) ≈ fd_a
    @test compute_Dv_update(fd_foo, 1.0, 1.0, 1.0) ≈ fd_a
    @test approximate_Dv(fd_foo, 1.0, 1.0, 1.0) ≈ 5
    @test compute_Dv(fd_foo, 1.0, (1.0, 1.0), (1.0, 1.0)) ≈ fd_a + fd_b
    @test compute_Dv_update(fd_foo, 1.0, (1.0, 1.0), (1.0, 1.0)) ≈ fd_a + fd_b
    @test approximate_Dv(fd_foo, 1.0, (1.0, 1.0), (1.0, 1.0)) ≈ 5 + 6
    @test compute_Dv(fd_foo, 5.0, (1.0, 1.0), (1.0, 1.0)) ≈ (fd_a + fd_b) * 5.0
    @test compute_Dv_update(fd_foo, 5.0, (1.0, 1.0), (1.0, 1.0)) ≈ (fd_a + fd_b) * 5.0
    @test approximate_Dv(fd_foo, 5.0, (1.0, 1.0), (1.0, 1.0)) ≈ (5 + 6) * 5.0
    @test compute_Dv(fd_foo, 1.0, (10.0, 5.0), (1.0, 1.0)) ≈ fd_a + fd_b
    @test compute_Dv_update(fd_foo, 1.0, (10.0, 5.0), (1.0, 1.0)) ≈ fd_a + fd_b
    @test approximate_Dv(fd_foo, 1.0, (10.0, 5.0), (1.0, 1.0)) ≈ 5 + 6
    @test compute_Dv(fd_foo, 1.0, (1.0, 1.0), (7.5, 6.3)) ≈ fd_a * 7.5 + fd_b * 6.3
    @test compute_Dv_update(fd_foo, 1.0, (1.0, 1.0), (7.5, 6.3)) ≈ fd_a * 7.5 + fd_b * 6.3
    @test approximate_Dv(fd_foo, 1.0, (1.0, 1.0), (7.5, 6.3)) ≈ 5 * 7.5 + 6 * 6.3
    @test compute_Dv(fd_foo, 5.0, (10.0, 5.0), (7.5, 6.3)) ≈ (fd_a * 7.5 + fd_b * 6.3) * 5
    @test compute_Dv_update(fd_foo, 5.0, (10.0, 5.0), (7.5, 6.3)) ≈ (fd_a * 7.5 + fd_b * 6.3) * 5
    @test approximate_Dv(fd_foo, 5.0, (10.0, 5.0), (7.5, 6.3)) ≈ (5 * 7.5 + 6 * 6.3) * 5

    # Check that finite differencing yields approximately correct results for vectors.
    N = 3
    z̄, x, y, vx, vy = ones.((N, N, N, N, N))
    @test compute_Dv(fd_foo, z̄, (x, y), (vx, vy)) ≈
        sum(z̄ .* ((fd_a * I) * vx)) + sum(z̄ .* ((fd_b * I) * vy))
    @test compute_Dv_update(fd_foo, z̄, (x, y), (vx, vy)) ≈
        sum(z̄ .* ((fd_a * I) * vx)) + sum(z̄ .* ((fd_b * I) * vy))
    @test approximate_Dv(fd_foo, z̄, (x, y), (vx, vy)) ≈
        sum(z̄ .* (UniformScaling(10) * vx)) + sum(z̄ .* (UniformScaling(11) * vy))
    z̄ = randn(N)
    @test compute_Dv(fd_foo, z̄, (x, y), (vx, vy)) ≈
        sum(z̄ .* ((fd_a * I) * vx)) + sum(z̄ .* ((fd_b * I) * vy))
    @test compute_Dv_update(fd_foo, z̄, (x, y), (vx, vy)) ≈
        sum(z̄ .* ((fd_a * I) * vx)) + sum(z̄ .* ((fd_b * I) * vy))
    @test approximate_Dv(fd_foo, z̄, (x, y), (vx, vy)) ≈
        sum(z̄ .* (UniformScaling(10) * vx)) + sum(z̄ .* (UniformScaling(11) * vy))
    v = randn(N)
    @test compute_Dv(fd_foo, z̄, (x, y), (vx, vy)) ≈
        sum(z̄ .* ((fd_a * I) * vx)) + sum(z̄ .* ((fd_b * I) * vy))
    @test compute_Dv_update(fd_foo, z̄, (x, y), (vx, vy)) ≈
        sum(z̄ .* ((fd_a * I) * vx)) + sum(z̄ .* ((fd_b * I) * vy))
    @test approximate_Dv(fd_foo, z̄, (x, y), (vx, vy)) ≈
        sum(z̄ .* (UniformScaling(10) * vx)) + sum(z̄ .* (UniformScaling(11) * vy))

    # Check that finite differencing yields approximately correct results for matrices.
    M, N = 3, 2
    Z̄, X, Y, VX, VY = ones.((M, M, M, M, M), (N, N, N, N, N))
    @test_broken approximate_Dv(fd_foo, Z̄, (X, Y), (VX, VY)) ≈
        compute_Dv(fd_foo, Z̄, (X, Y), (VX, VY))
    @test compute_Dv(fd_foo, Z̄, (X, Y), (VX, VY)) ≈
        compute_Dv_update(fd_foo, Z̄, (X, Y), (VX, VY))
    Ȳ = randn(M, N)
    @test_broken approximate_Dv(fd_foo, Z̄, (X, Y), (VX, VY)) ≈
        compute_Dv(fd_foo, Z̄, (X, Y), (VX, VY))
    @test compute_Dv(fd_foo, Z̄, (X, Y), (VX, VY)) ≈
        compute_Dv_update(fd_foo, Z̄, (X, Y), (VX, VY))
end
