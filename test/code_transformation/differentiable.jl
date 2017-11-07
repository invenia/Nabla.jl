@testset "code_transformation/differentiable" begin

    import Nabla.Nabla: unionise_type, unionise_arg, unionise_eval, unionise_macro_eval,
        unionise_sig, unionise

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

    # Test Nabla.unionise_eval.
    @test unionise_eval(:(eval(:foo))) == :(eval(:(@unionise foo)))
    @test unionise_eval(:(eval(DiffBase, :foo))) == :(eval(DiffBase, :(@unionise foo)))
    @test unionise_eval(:(eval(:(println("foo"))))) == :(eval(:(@unionise println("foo"))))
    @test unionise_eval(:(eval(DiffBase, :(println("foo"))))) ==
        :(eval(DiffBase, :(@unionise println("foo"))))

    # Test Nabla.unionise_macro_eval.
    @test unionise_macro_eval(:(@eval foo)) == :(@eval @unionise foo)
    @test unionise_macro_eval(:(@eval DiffBase foo)) == :(@eval DiffBase @unionise foo)
    @test unionise_macro_eval(:(@eval println("foo"))) == :(@eval @unionise println("foo"))
    @test unionise_macro_eval(:(@eval DiffBase println("foo"))) ==
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
    @test unionise(:(@eval foo)) == unionise_macro_eval(:(@eval foo))
    @test unionise(:(@eval DiffBase foo)) == unionise_macro_eval(:(@eval DiffBase foo))
end
