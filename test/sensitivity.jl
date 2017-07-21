@testset "sensitivity_tests" begin

    import Base.Meta.quot

    # # "Test" `DiffCore.get_body`. (Not currently unit testing this as it is awkward. Will
    # # change this at some point in the future to be more unit-testable.)
    # let
    #     println()
    #     from_func = DiffCore.get_body(:bar, :(Base.bar), :(Tuple{Real}), [:x])
    #     full_expr = DiffCore.add_intercept(:bar, :(Base.bar), :(Tuple{Real}))
    #     println(full_expr)
    # end
    # let
    #     println()
    #     from_func = DiffCore.get_body(:bar, :(Base.bar), :(Tuple{T} where T<:Real), [:x])
    #     full_expr = DiffCore.add_intercept(:bar, :(Base.bar), :(Tuple{T} where T<:Real))
    #     println(full_expr)
    # end
    # let
    #     println()
    #     from_func = DiffCore.get_body(:bar, :(Base.bar), :(Tuple{Vararg}), [:x])
    #     full_expr = DiffCore.add_intercept(:bar, :(Base.bar), :(Tuple{Vararg}))
    #     println(full_expr)
    # end

    # Test `DiffCore.branch_expr`.
    import Nabla.DiffCore.branch_expr
    let
        from_func = branch_expr(:bar, [true], (Int,), (:x,), :((x,)))
        tape = Expr(:call, :getfield, :x, quot(:tape))
        expected = Expr(:call, :Branch, :bar, :((x,)), tape)
        @test from_func == expected
    end
    let
        from_func = branch_expr(:bar, [true, false], (Int,), (:x, :y), :((x, y)))
        tape = Expr(:call, :getfield, :x, quot(:tape))
        expected = Expr(:call, :Branch, :bar, :((x, y)), tape)
        @test from_func == expected
    end
    let
        from_func = branch_expr(:bar, [false, true], (Int, Int), (:x, :y), :((x, y)))
        tape = Expr(:call, :getfield, :y, quot(:tape))
        expected = Expr(:call, :Branch, :bar, :((x, y)), tape)
        @test from_func == expected
    end
    let
        from_func = branch_expr(:bar, [true, true], (Int, Int), (:x, :y), :((x, y)))
        tape = Expr(:call, :getfield, :x, quot(:tape))
        expected = Expr(:call, :Branch, :bar, :((x, y)), tape)
        @test from_func == expected
    end
    let
        from_func = branch_expr(:bar, [true], ((Leaf{Float64},),), (:x,), :((x...)))
        tape = Expr(:call, :getfield, Expr(:ref, :x, :1), quot(:tape))
        expected = Expr(:call, :Branch, :bar, :((x...,)), tape)
        @test from_func == expected
    end

    # Test `DiffCore.get_union_call`.
    import Nabla.DiffCore.get_union_call
    let
        from_func = get_union_call(:goo, :(Tuple{Real}))[1]
        expected = :(goo(x1::Union{Real, Node{<:Real}}))
        @test from_func == expected
    end
    let
        from_func = get_union_call(:goo, :(Tuple{Real, Number}))[1]
        expected = :(
        goo(
            x1::Union{Real, Node{<:Real}},
            x2::Union{Number, Node{<:Number}},
        ))
        @test from_func == expected
    end
    let
        arg = :(AbstractArray{V} where V)
        from_func = get_union_call(:goo, :(Tuple{$arg}))[1]
        expected = :(goo(x1::Union{$arg, Node{<:($arg)}}))
        @test from_func == expected
    end
    let 
        from_func = get_union_call(:goo, :(Tuple{V, V} where V<:Real))[1]
        expected = :(goo(x1::Union{V, Node{<:V}}, x2::Union{V, Node{<:V}}) where V<:Real)
        @test from_func == expected
    end
    let
        from_func = get_union_call(:goo, :(Tuple{Vararg{Real}}))[1]
        expected = :(goo(x1::Vararg{Union{Real, Node{<:Real}}}))
        @test from_func == expected
    end
    let
        from_func = get_union_call(:goo, :(Tuple{Number, Vararg{Real}}))[1]
        expected = :(
        goo(
            x1::Union{Number, Node{<:Number}},
            x2::Vararg{Union{Real, Node{<:Real}}},
        ))
        @test from_func == expected
    end
    let
        from_func = get_union_call(:goo, :(Tuple{Vararg{V where V<:Real}}))[1]
        expected = :(goo(x1::Vararg{Union{V where V<:Real, Node{<:(V where V<:Real)}}}))
        @test from_func == expected
    end
    let
        arg_type = :(Union{Real, AbstractArray})
        from_func = get_union_call(:goo, :(Tuple{Vararg{$arg_type}}))[1]
        expected = :(goo(x1::Vararg{Union{$arg_type, Node{<:$arg_type}}}))
        @test from_func == expected
    end
    let
        arg_type = :(Union{Real, AbstractArray{V} where V<:Real})
        from_func = get_union_call(:goo, :(Tuple{Vararg{$arg_type}}))[1]
        expected = :(goo(x1::Vararg{Union{$arg_type, Node{<:$arg_type}}}))
        @test from_func == expected
    end

    # Test `DiffCore.bypass_diff_expr`.
    let
        from_func = DiffCore.bypass_diff_expr(:goo, (Float64,), (:x,))
        expected = :(DiffCore.call_with_originals(goo, x))
        @test from_func == expected
    end
    let
        from_func = DiffCore.bypass_diff_expr(:goo, (Float64, Float32), (:x, :y))
        expected = :(DiffCore.call_with_originals(goo, x, y))
        @test from_func == expected
    end
    let
        from_func = DiffCore.bypass_diff_expr(:goo, ((Float64, Float64),), (:x,))
        expected = :(DiffCore.call_with_originals(goo, x...))
        @test from_func == expected
    end
    let
        from_func = DiffCore.bypass_diff_expr(:goo, (Float64, (Float32, Int)), (:x, :y))
        expected = :(DiffCore.call_with_originals(goo, x, y...))
        @test from_func == expected
    end
end
