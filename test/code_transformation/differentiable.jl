# NOTE: As of https://github.com/JuliaLang/julia/pull/21746, all macro calls have
# a LineNumberNode. Since our functions that do expression transformation don't have
# access to the source code locations, we don't insert a LineNumberNode, which means
# that a transformed expression won't compare equal to an expression literal due to
# line information. Thus for testing purposes we define a function that removes line
# information from expressions as well as a custom equality operator that uses this.
# Note that it's only necessary for comparisons that deal with macro calls.
function skip_line_info(ex::Expr)
    map!(arg->arg isa LineNumberNode ? nothing : skip_line_info(arg), ex.args, ex.args)
    ex
end
skip_line_info(ex) = ex
≃(a, b) = skip_line_info(a) == skip_line_info(b)

@testset "code_transformation/differentiable" begin

    import Nabla.Nabla: unionise_type, unionise_arg, unionise_subtype, unionise_eval,
        unionise_macro_eval, unionise_sig, unionise, unionise_struct

    # Test Nabla.unionise_arg. This depends upon Nabla.unionise_type, so we express
    # our tests in terms of this.
    @test unionise_arg(:x) == :x
    @test unionise_arg(:foo) == :foo
    @test unionise_arg(:(::x)) == :(::$(unionise_type(:x)))
    @test unionise_arg(:(::T{V})) == :(::$(unionise_type(:(T{V}))))
    @test unionise_arg(:(x::T)) == :(x::$(unionise_type(:T)))
    @test unionise_arg(:(x::T{V})) == :(x::$(unionise_type(:(T{V}))))
    @test unionise_arg(:(::T{V} where V)) == :(::$(unionise_type(:(T{V} where V))))
    @test unionise_arg(:(::T{V} where V<:Q)) == :(::$(unionise_type(:(T{V} where V<:Q))))
    @test unionise_arg(:(x::T{V} where V<:Q)) == :(x::$(unionise_type(:(T{V} where V<:Q))))

    # Test unionise_arg for varargs case.
    let dots = Symbol("...")
        @test unionise_arg(Expr(dots, :x)) == Expr(dots, :x)
        @test unionise_arg(Expr(dots, :(x::T))) == Expr(dots, unionise_arg(:(x::T)))
    end

    # Test unionise_subtype.
    @test unionise_subtype(:T) == :T
    @test unionise_subtype(:(T<:V)) == :(T<:$(unionise_type(:V)))

    # Test Nabla.unionise_eval.
    @test unionise_eval(:(eval(:foo))) ≃ :(eval(:(@unionise foo)))
    @test unionise_eval(:(eval(DiffBase, :foo))) ≃ :(eval(DiffBase, :(@unionise foo)))
    @test unionise_eval(:(eval(:(println("foo"))))) ≃ :(eval(:(@unionise println("foo"))))
    @test unionise_eval(:(eval(DiffBase, :(println("foo"))))) ≃
        :(eval(DiffBase, :(@unionise println("foo"))))

    # Test Nabla.unionise_macro_eval.
    @test unionise_macro_eval(:(@eval foo)) ≃ :(@eval @unionise foo)
    @test unionise_macro_eval(:(@eval DiffBase foo)) ≃ :(@eval DiffBase @unionise foo)
    @test unionise_macro_eval(:(@eval println("foo"))) ≃ :(@eval @unionise println("foo"))
    @test unionise_macro_eval(:(@eval DiffBase println("foo"))) ≃
        :(@eval DiffBase @unionise println("foo"))

    # Test Nabla.unionise. This depends upon Nabla.unionise_arg, so we express
    # our tests in terms of this function.
    @test unionise_sig(:((x,))) == :((x,))
    @test unionise_sig(:((x, y))) == :((x, y))
    @test unionise_sig(:(foo(x))) == :(foo(x))
    @test unionise_sig(:(foo(x, y))) == :(foo(x, y))
    @test unionise_sig(:((x::T,))) == :(($(unionise_arg(:(x::T))),))
    @test unionise_sig(:(foo(x::T))) == :(foo($(unionise_arg(:(x::T)))))
    @test unionise_sig(:(foo(x::T) where T)) == :(foo($(unionise_arg(:(x::T)))) where T)

    @test isequal(  # special case for a redudant where N in a Vararg
        Nabla.unionise_sig(:(x2::(Vararg{Int64, N} where N))),
        :(x2::Vararg{Union{Int64, Node{<:Int64}}}),
    )


    # Test Nabla.unionise_struct. Written in terms of Nabla.unionise_arg.
    @test unionise_struct(:(struct Foo end)) == :(struct Foo end)
    @test unionise_struct(:(struct Foo{T} end)) ==
        :(struct Foo{$(unionise_subtype(:T))} end)
    @test unionise_struct(:(struct Foo{T<:Real} end)) ==
        :(struct Foo{$(unionise_subtype(:(T<:Real)))} end)
    @test unionise_struct(:(struct Foo{T<:V, Q<:U} end)) ==
        :(struct Foo{$(unionise_subtype(:(T<:V))), $(unionise_subtype(:(Q<:U)))} end)
    @test unionise_struct(:(struct Foo <: Bar end)) == :(struct Foo <: Bar end)
    @test unionise_struct(:(struct Foo{T} <: Bar end)) == :(struct Foo{T} <: Bar end)
    @test unionise_struct(:(struct Foo{T<:V} <: Bar end)) ==
        :(struct Foo{$(unionise_subtype(:(T<:V)))} <: Bar end)
    @test unionise_struct(:(struct Foo{T<:V, Q<:U} <: Bar end)) ==
        :(struct Foo{$(unionise_subtype(:(T<:V))), $(unionise_subtype(:(Q<:U)))} <: Bar end)

    # Test Nabla.make_accept_nodes. Heavily depends upon Nabla.unionise_sig, so we
    # express tests in terms of this function.
    @test unionise(:hi) == :hi
    @test unionise(:(N = 5)) == :(N = 5)
    @test unionise(:(N, M = 5, 4)) == :(N, M = 5, 4)
    @test unionise(:((N, M) = (5, 4))) == :((N, M) = (5, 4))
    @test unionise(:((N, M) = 5, 4)) == :((N, M) = 5, 4)
    @test unionise(:(N, M = (5, 4))) == :(N, M = (5, 4))
    @test unionise(Expr(Symbol("->"), :x, :x)) == Expr(Symbol("->"), :x, :x)
    @test unionise(Expr(Symbol("->"), :(x::Real), :(5x))) ==
        Expr(Symbol("->"), unionise_sig(:(x::Real)), :(5x))
    @test unionise(Expr(Symbol("->"), :((x::Real,)), :(5x))) ==
        Expr(Symbol("->"), unionise_sig(:((x::Real,))), :(5x))
    @test unionise(Expr(Symbol("->"), :((x::Real, y::Real)), :(5x))) ==
        Expr(Symbol("->"), unionise_sig(:((x::Real, y::Real))), :(5x))
    @test unionise(Expr(:function, :((x,)), :(return x))) ==
        Expr(:function, unionise_sig(:((x,))), :(return x))
    @test unionise(Expr(:function, :((x::Real,)), :(return x))) ==
        Expr(:function, unionise_sig(:((x::Real,))), :(return x))
    @test unionise(Expr(:function, :(foo(x)), :(return x))) ==
        Expr(:function, unionise_sig(:(foo(x))), :(return x))
    @test unionise(Expr(:function, :(foo(x::Real)), :(return x))) ==
        Expr(:function, unionise_sig(:(foo(x::Real))), :(return x))
    @test unionise(Expr(Symbol("="), :(foo(x)), :x)) ==
        Expr(Symbol("="), unionise_sig(:(foo(x))), :x)
    @test unionise(Expr(Symbol("="), :(foo(x::Real)), :x)) ==
        Expr(Symbol("="), unionise_sig(:(foo(x::Real))), :x)
    @test unionise(Expr(Symbol("="), :(foo(x::T, y::T) where T), :x)) ==
        Expr(Symbol("="), unionise_sig(:(foo(x::T, y::T) where T)), :x)
    @test unionise(:(eval(:foo))) == unionise_eval(:(eval(:foo)))
    @test unionise(:(eval(DiffBase, :foo))) == unionise_eval(:(eval(DiffBase, :foo)))
    @test unionise(:(@eval foo)) ≃ unionise_macro_eval(:(@eval foo))
    @test unionise(:(@eval DiffBase foo)) ≃ unionise_macro_eval(:(@eval DiffBase foo))
    @test unionise(:(struct Foo{T<:V} end)) == unionise_struct(:(struct Foo{T<:V} end))

    # @unionise with default values and keywords
    UT = unionise_type(:T)
    raw = unionise(:(f(x::T, y::T=2; z::T=4) = x + y + z))
    new = :(f(x::$UT, y::$UT=2; z::T=4) = x + y + z)
    @test raw ≃ new

    # @unionise error conditions
    @test_throws ArgumentError unionise(:(f(@nospecialize x) = x))
end
