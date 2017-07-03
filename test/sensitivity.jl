@testset "sensitivity" begin

    function check_sensitivity_func()
        x_ = randn(5)
        x = Root(x_, Tape())
        y = sum(x)
        ∇y = AutoGrad2.∇(y)
        return all(∇y[x] == ones(x_))
    end
    @test check_sensitivity_func()

    # Binary function with both sensitivities defined.
    foo(x::Real, y::Real) = x + y
    AutoGrad2.∇(::typeof(foo), ::Type{Arg{1}}, p, x::Real, y::Real, z::Real, z̄::Real) = z̄
    AutoGrad2.∇(::typeof(foo), ::Type{Arg{2}}, p, x::Real, y::Real, z::Real, z̄::Real) = z̄

    # Binary function without sensitivity definitions.
    goo(x::Real, y::Real) = x * y

    # Check that foo's ∇ methods work as expected, but goo's throw an error.
    @test ∇(foo, Arg{1}, (), 5.0, 4.0, 3.0, 2.0) == 2.0
    @test ∇(foo, Arg{2}, (), 5.0, 4.0, 3.0, 2.0) == 2.0
    @test_throws MethodError ∇(foo, Arg{3}, (), 5.0, 4.0, 3.0, 2.0)
    @test_throws MethodError ∇(goo, Arg{1}, (), 5.0, 4.0, 3.0, 2.0)
    @test_throws MethodError ∇(goo, Arg{2}, (), 5.0, 4.0, 3.0, 2.0)
end
