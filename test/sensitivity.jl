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

# function get_union_call(foo::Symbol, type_tuple::Expr)

#     # Get type info from tuple and declare a collection of symbols for use in the call.
#     types = get_types(get_body(type_tuple))
#     arg_names, arg_types = [gensym() for _ in types], [gensym() for _ in types]
#     arg_names = [Symbol("x$j") for j in 1:length(types)]

#     # Remove strip out Vararg stuff, compute unioned types, and re-add Vararg stuff.
#     type_info = remove_vararg.(types)
#     unioned_types = [:(Union{$typ, Node{$par} where $par <: $typ})
#         for (typ, par) in zip(types, arg_types)]
#     vararged_types = replace_vararg.(unioned_types, type_info)

#     # Generate the call.
#     typed_args = [:($name::$typ) for (name, typ) in zip(arg_names, vararged_types)]
#     # subtyped_pars = [:($par_typ <: $(typ[1])) for (par_typ, typ) in zip(arg_types, type_info)]
#     # new_body = Expr(:where, Expr(:call, foo, typed_args...), subtyped_pars...)
#     new_body = Expr(:call, foo, typed_args...)
#     return replace_body(type_tuple, new_body), arg_names
# end

    # Test `DiffCore.get_union_call`.
    import Nabla.DiffCore.get_union_call
    let
        from_func = get_union_call(:goo, :(Tuple{Real}), [:T])[1]
        expected = :(goo(x1::Union{Real, Node{T} where T<:Real}))
        @test from_func == expected
    end
    let
        from_func = get_union_call(:goo, :(Tuple{Real, Number}), [:T, :T])[1]
        expected = :(
        goo(
            x1::Union{Real, Node{T} where T<:Real},
            x2::Union{Number, Node{T} where T<:Number},
        ))
        @test from_func == expected
    end
    let
        arg = :(AbstractArray{V} where V)
        from_func = get_union_call(:goo, :(Tuple{$arg}), [:T])[1]
        expected = :(goo(x1::Union{$arg, Node{T} where T<:($arg)}))
        @test from_func == expected
    end
    let 
        from_func = get_union_call(:goo, :(Tuple{V, V} where V<:Real), [:T, :T])[1]
        expected = :(goo(x1::Union{V, Node{T} where T<:V}, x2::Union{V, Node{T} where T<:V}) where V<:Real)
        @test from_func == expected
    end
    let
        from_func = get_union_call(:goo, :(Tuple{Vararg{Real}}), [:T])[1]
        expected = :(goo(x1::Vararg{Union{Real, Node{T} where T<:Real}}))
        @test from_func == expected
    end
    let
        from_func = get_union_call(:goo, :(Tuple{Number, Vararg{Real}}), [:T, :T])[1]
        expected = :(
        goo(
            x1::Union{Number, Node{T} where T<:Number},
            x2::Vararg{Union{Real, Node{T} where T<:Real}},
        ))
        @test from_func == expected
    end
    let
        from_func = get_union_call(:goo, :(Tuple{Vararg{V} where V<:Real}), [:T])[1]
        expected = :(goo(x1::Vararg{Union{V where V<:Real, Node{T} where T<:(V where V<:Real)}}))
        @test from_func == expected
    end
    let
        arg_type = :(Union{Real, AbstractArray})
        from_func = get_union_call(:goo, :(Tuple{Vararg{$arg_type}}), [:T])[1]
        expected = :(goo(x1::Vararg{Union{$arg_type, Node{T} where T<:$arg_type}}))
        @test from_func == expected
    end
    let
        arg_type = :(Union{Real, AbstractArray{V} where V<:Real})
        from_func = get_union_call(:goo, :(Tuple{Vararg{$arg_type}}), [:G])[1]
        expected = :(goo(x1::Vararg{Union{$arg_type, Node{G} where G<:$arg_type}}))
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

    # Test DiffCore.replace_body.
    @test DiffCore.replace_body(:Real, :Float64) == :Float64
    @test DiffCore.replace_body(:Real, :(Union{Float64, foo})) == :(Union{Float64, foo})
    @test DiffCore.replace_body(:(T where T), :(Node{T})) == :(Node{T} where T)
    @test DiffCore.replace_body(:(T where T <: A{V} where V), :(Node{T})) ==
        :(Node{T} where T <: A{V} where V)

    # Test DiffCore.get_body.
    @test DiffCore.get_body(:Real) == :Real
    @test DiffCore.get_body(:(Node{T})) == :(Node{T})
    @test DiffCore.get_body(:(Node{T} where T)) == :(Node{T})
    @test DiffCore.get_body(:(Node{T} where T<:A{Q} where Q)) == :(Node{T})
    @test DiffCore.get_body(:(Node{A{T} where T})) == :(Node{A{T} where T})
    @test DiffCore.get_body(:(Node{A{T} where T} where A)) == :(Node{A{T} where T})

    # Test DiffCore.isa_vararg.
    @test DiffCore.isa_vararg(:foo) == false
    @test DiffCore.isa_vararg(:Vararg) == true
    @test DiffCore.isa_vararg(:(Node{T} where T)) == false
    @test DiffCore.isa_vararg(:(Vararg{T} where T)) == true
    @test DiffCore.isa_vararg(:(Vararg{T, N} where T where N)) == true

    # Test DiffCore.remove_vararg.
    @test DiffCore.remove_vararg(:Vararg) == (:Any, :Vararg)
    @test DiffCore.remove_vararg(:(Node{T})) == (:(Node{T}), :nothing)
    @test DiffCore.remove_vararg.([:Vararg]) == [(:Any, :Vararg)]
    @test DiffCore.remove_vararg.(:Vararg) == (:Any, :Vararg)
    @test DiffCore.remove_vararg.([:Vararg, :Vararg]) == [(:Any, :Vararg), (:Any, :Vararg)]
    @test DiffCore.remove_vararg(:Real) == (:Real, :nothing)
    @test DiffCore.remove_vararg(:(Vararg{T} where T)) == (:(T where T), :Vararg)
    @test DiffCore.remove_vararg(:(Vararg{T, N} where T<:Real)) == (:(T where T<:Real), :N)

    # Test DiffCore.replace_vararg.
    @test DiffCore.replace_vararg(:(U{T, N{T}}), (:V, :nothing)) == :(U{T, N{T}})
    @test DiffCore.replace_vararg(:(Real), (:Any, :Vararg)) == :(Vararg{Real})
    @test DiffCore.replace_vararg(:(T where T), (:T, :Vararg)) == :(Vararg{T} where T)
    @test DiffCore.replace_vararg(:(T where T), (:T, :2)) == :(Vararg{T, 2} where T)
    @test DiffCore.replace_vararg(:(U{T} where T), (:T, :N)) == :(Vararg{U{T}, N} where T)
    @test DiffCore.replace_vararg(:(U{T} where N), (:T, :N)) == :(Vararg{U{T}, N} where N)
end
