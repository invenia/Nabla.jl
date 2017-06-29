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
    AutoGrad2.∇(::Type{Arg{1}}, ::typeof(foo), p, x::Real, y::Real, z::Real, z̄::Real) = z̄
    AutoGrad2.∇(::Type{Arg{2}}, ::typeof(foo), p, x::Real, y::Real, z::Real, z̄::Real) = z̄

    # Binary function without sensitivity definitions.
    goo(x::Real, y::Real) = x * y

    # Check that foo's ∇ methods work as expected, but goo's throw an error.
    @test ∇(Arg{1}, foo, (), 5.0, 4.0, 3.0, 2.0) == 2.0
    @test ∇(Arg{2}, foo, (), 5.0, 4.0, 3.0, 2.0) == 2.0
    @test_throws MethodError ∇(Arg{3}, foo, (), 5.0, 4.0, 3.0, 2.0)
    @test_throws MethodError ∇(Arg{1}, goo, (), 5.0, 4.0, 3.0, 2.0)
    @test_throws MethodError ∇(Arg{2}, goo, (), 5.0, 4.0, 3.0, 2.0)

    # Does having exchanged the old body for the new result in the expected body?
    function check_change_unionall_body(u, new_body, expected)
        return change_unionall_body(u, new_body) == expected
    end

    @test check_change_unionall_body(Tuple{Real}, Tuple{Number}, Tuple{Number})
    @test check_change_unionall_body(
        Tuple{T} where T,
        Tuple{Q} where Q<:T,
        Tuple{Q} where Q<:T where T
    )

end
