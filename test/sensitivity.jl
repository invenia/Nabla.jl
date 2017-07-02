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

    function check_get_tape(syms, diffs, expected)
        return AutoGrad2.get_tape(syms, diffs) == expected
    end
    @test_throws ArgumentError AutoGrad2.get_tape([:x], [false])
    @test_throws ArgumentError AutoGrad2.get_tape([:x, :y], [false, false])
    @test_throws ArgumentError AutoGrad2.get_tape([:x, :y, :z], [false, false, false])
    @test check_get_tape([:x], [true], Expr(Symbol("."), :x, :(:tape)))
    @test check_get_tape([:x, :y], [true, true], Expr(Symbol("."), :x, :(:tape)))
    @test check_get_tape([:x, :y], [true, false], Expr(Symbol("."), :x, :(:tape)))
    @test check_get_tape([:x, :y], [false, true], Expr(Symbol("."), :y, :(:tape)))

    function check_curly_to_where(expr, expected)
        return AutoGrad2.curly_to_where(expr) == expected
    end
    @test_throws ErrorException AutoGrad2.curly_to_where(:(foo(x)))
    @test_throws ErrorException AutoGrad2.curly_to_where(:(foo(x::Real)))
    @test_throws ErrorException AutoGrad2.curly_to_where(:(foo(x::T) where T))
    @test check_curly_to_where(:(foo{T}(x::T)), :(foo(x::T) where T))
    @test check_curly_to_where(:(foo{T, V}(x::T, y::V)), :(foo(x::T, y::V) where {T, V}))
    @test check_curly_to_where(:(foo{T<:Real}(x::T)), :(foo(x::T) where T<:Real))

    function check_call_from_where(expr, expected)
        return AutoGrad2.call_from_where(expr) == expected
    end
    @test_throws ErrorException AutoGrad2.call_from_where(:boo)
    @test_throws ErrorException AutoGrad2.call_from_where(:(T where T))
    @test_throws ErrorException AutoGrad2.call_from_where(:(Tuple{T}))
    @test check_call_from_where(:(foo(x::T) where T), :(foo(x::T)))
    @test check_call_from_where(:(foo(x::T, y::Real) where T<:Real), :(foo(x::T, y::Real)))
    @test check_call_from_where(:(foo(x::T, y::V) where T where V), :(foo(x::T, y::V)))
    @test check_call_from_where(:(foo(x::T, y::V) where {T, V}), :(foo(x::T, y::V)))

    function check_change_where_call(expr, new_call, expected)
        return AutoGrad2.change_where_call(expr, new_call) == expected
    end
    @test_throws ErrorException AutoGrad2.change_where_call(:(Tuple{T}), :boo)
    @test_throws ErrorException AutoGrad2.change_where_call(:(T where T), :boo)
    @test check_change_where_call(:(foo(x)), :boo, :boo)
    @test check_change_where_call(:(foo(x::Any)), :boo, :boo)
    @test check_change_where_call(:(foo(x::T) where T), :boo, :(boo where T))
    @test check_change_where_call(
        :(foo(x::T{V}) where V<:Real where T<:Node),
        :boo,
        :(boo where V<:Real where T<:Node),
    )
    @test check_change_where_call(
        :(foo(x::T{V}) where {V<:Real, T<:Node}),
        :boo,
        :(boo where {V<:Real, T<:Node}),
    )


    function check_to_tuple_type(expr, expected)
        return AutoGrad2.to_tuple_type(expr) == expected
    end
    @test check_to_tuple_type(:(foo(x)), :(Tuple{typeof(foo), Any}))
    @test check_to_tuple_type(:(foo(x::Any)), :(Tuple{typeof(foo), Any}))
    @test check_to_tuple_type(:(foo(x::Real)), :(Tuple{typeof(foo), Real}))
    @test check_to_tuple_type(:(foo(x::T) where T), :(Tuple{typeof(foo), T} where T))
    @test check_to_tuple_type(
        :(foo(x::T) where T<:Number),
        :(Tuple{typeof(foo), T} where T<:Number),
    )
    @test check_to_tuple_type(
        :(foo(x::Real, y::T) where T<:Real),
        :(Tuple{typeof(foo), Real, T} where T<:Real),
    )
    @test check_to_tuple_type(
        :(foo(x::Union{Real, String})),
        :(Tuple{typeof(foo), Union{Real, String}}),
    )
    @test check_to_tuple_type(
        :(foo(x::Union{T, Real}, y) where T),
        :(Tuple{typeof(foo), Union{T, Real}, Any} where T),
    )
    @test check_to_tuple_type(
        :(foo(x::T where T)),
        :(Tuple{typeof(foo), T where T}),
    )
    @test check_to_tuple_type(
        :(foo(x::Union{T, Node{T}} where T)),
        :(Tuple{typeof(foo), Union{T, Node{T}} where T}),
    )
    @test check_to_tuple_type(
        :(foo(x::Union{T, Node{T}}, y::Union{T, Node{T}}) where T<:Real),
        :(Tuple{typeof(foo), Union{T, Node{T}}, Union{T, Node{T}}} where T<:Real),
    )
end
