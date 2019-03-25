using Base.Meta
using Nabla: boxed_method

function expected_func(sig::Expr, body::Expr)
    return Expr(:function, sig, Expr(:block, LineNumberNode(0), body))
end

@testset "sensitivity" begin
    # # "Test" `Nabla.get_body`. (Not currently unit testing this as it is awkward. Will
    # # change this at some point in the future to be more unit-testable.)
    # let
    #     println()
    #     from_func = Nabla.get_body(:bar, :(Base.bar), :(Tuple{Real}), [:x])
    #     full_expr = Nabla.add_intercept(:bar, :(Base.bar), :(Tuple{Real}))
    #     println(full_expr)
    # end
    # let
    #     println()
    #     from_func = Nabla.get_body(:bar, :(Base.bar), :(Tuple{T} where T<:Real), [:x])
    #     full_expr = Nabla.add_intercept(:bar, :(Base.bar), :(Tuple{T} where T<:Real))
    #     println(full_expr)
    # end
    # let
    #     println()
    #     from_func = Nabla.get_body(:bar, :(Base.bar), :(Tuple{Vararg}), [:x])
    #     full_expr = Nabla.add_intercept(:bar, :(Base.bar), :(Tuple{Vararg}))
    #     println(full_expr)
    # end

    @testset "boxed_method" begin
        let
            from_func = boxed_method(:foo, :(Tuple{Any}), [true], [:x1])
            expected = expected_func(:(foo(x1::Node{<:Any})),
                                     :(Branch(foo, (x1,), getfield(x1, $(quot(:tape))))))
            @test from_func == expected
        end
        let
            from_func = boxed_method(:foo, :(Tuple{T{V}}), [true], [:x1])
            expected = expected_func(:(foo(x1::Node{<:T{V}})),
                                     :(Branch(foo, (x1,), getfield(x1, $(quot(:tape))))))
            @test from_func == expected
        end
        let
            from_func = boxed_method(:foo, :(Tuple{Any, Any}), [true, false], [:x1, :x2])
            expected = expected_func(:(foo(x1::Node{<:Any}, x2::Any)),
                                     :(Branch(foo, (x1, x2), getfield(x1, $(quot(:tape))))))
            @test from_func == expected
        end
        let
            from_func = boxed_method(:foo, :(Tuple{Any, Any}), [true, true], [:x1, :x2])
            expected = expected_func(:(foo(x1::Node{<:Any}, x2::Node{<:Any})),
                                     :(Branch(foo, (x1, x2), getfield(x1, $(quot(:tape))))))
            @test from_func == expected
        end
        let
            from_func = boxed_method(:foo, :(Tuple{Any, Any}), [false, true], [:x1, :x2])
            expected = expected_func(:(foo(x1::Any, x2::Node{<:Any})),
                                     :(Branch(foo, (x1, x2), getfield(x2, $(quot(:tape))))))
            @test from_func == expected
        end
        let
            from_func = boxed_method(:foo, :(Tuple{T} where T), [true], [:x1])
            expected = expected_func(:(foo(x1::Node{<:T}) where T),
                                     :(Branch(foo, (x1,), getfield(x1, $(quot(:tape))))))
            @test from_func == expected
        end
        let
            from_func = boxed_method(:foo, :(Tuple{Any, Any}), [false, true], [:x1, :x2]; a=1, b=2)
            expected = Expr(:block,
                            expected_func(:(foo(x1::Any, x2::Node{<:Any}; a=1, b=2)),
                                          :(_foo(x1, x2, a, b))),
                            expected_func(:(_foo(x1::Any, x2::Node{<:Any}, a::Any, b::Any)),
                                          :(Branch(_foo, (x1, x2, a, b), getfield(x2, $(quot(:tape)))))))
            @test from_func == expected
        end
    end

    # Test `Nabla.branch_expr`.
    import Nabla.Nabla.branch_expr
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
        from_func = branch_expr(:bar, [true], ((Leaf{Float64},),), (:x,), :((x...,)))
        tape = Expr(:call, :getfield, Expr(:ref, :x, :1), quot(:tape))
        expected = Expr(:call, :Branch, :bar, :((x...,)), tape)
        @test from_func == expected
    end

    # Test Nabla.invoke_expr
    import Nabla.invoke_expr
    let
        from_func = invoke_expr(:foo, :(Tuple{T, V}), [:x1, :x2])
        expected = Expr(:call, :invoke, :foo, :(Tuple{T, V}), :x1, :x2)
        @test from_func == expected
    end

    # Test `Nabla.get_union_call`.
    import Nabla.Nabla.get_union_call
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

    using Nabla: kwfname, kwsort!, kwsortby
    @testset "keyword utilities" begin
        @test kwfname(sin) === :_sin
        @test kwfname(:sin) === :_sin

        ex = Expr(:parameters, Expr(:..., :kwargs), Expr(:kw, :x, 2), Expr(:kw, :a, 1))
        kwsort!(ex)
        @test ex == Expr(:parameters, Expr(:kw, :a, 1), Expr(:kw, :x, 2), Expr(:..., :kwargs))
        ex = Expr(:tuple, Expr(:(=), :a, 1), Expr(:..., :kwargs), Expr(:(=), :x, 2))
        kwsort!(ex)
        @test ex == Expr(:tuple, Expr(:(=), :a, 1), Expr(:(=), :x, 2), Expr(:..., :kwargs))
        nt = (b="b", c="c", a=22)
        @test kwsort!(nt) == (a=22, b="b", c="c")
        @test kwsort!(NamedTuple()) == NamedTuple()
        @test collect(kwsort!(pairs(nt))) == [:a => 22, :b => "b", :c => "c"]

        @test_throws ArgumentError kwsort!(:(f(x)))
        @test_throws ArgumentError kwsortby(:(f(x)))
    end
end
