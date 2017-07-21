@testset "code_transformation/differentiable" begin

    import Nabla.DiffCore: unionise_type, unionise_arg, unionise_sig, unionise

    # Test DiffCore.unionise_arg. This depends upon DiffCore.unionise_type, so we express
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

    # Test DiffCore.unionise. This depends upon DiffCore.unionise_arg, so we express
    # our tests in terms of this function.
    @test unionise_sig(:((x,))) == :((x,))
    @test unionise_sig(:((x, y))) == :((x, y))
    @test unionise_sig(:(foo(x))) == :(foo(x))
    @test unionise_sig(:(foo(x, y))) == :(foo(x, y))
    @test unionise_sig(:((x::T,))) == :(($(unionise_arg(:(x::T))),))
    @test unionise_sig(:(foo(x::T))) == :(foo($(unionise_arg(:(x::T)))))

    # Test DiffCore.make_accept_nodes. Heavily depends upon DiffCore.unionise_sig, so we
    # express tests in terms of this function.
    @test unionise(:hi) == :hi
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


    # println(make_accept_nodes(quote

    #     function(x)
    #         return 5x
    #     end
    #     function(x, y)
    #         return x + y
    #     end
    #     function(x::Real)
    #         return 10x
    #     end
    #     function(x::Real, y::Float64)
    #         return 10x + 5y
    #     end

    #     x->x
    #     (x)->x
    #     (x,)->x
    #     (x::Real)->x

    #     function foo(x)
    #         return x
    #     end
    #     function foo(x::Real)
    #         return x
    #     end
    #     function foo(x::Real, y::Real, z::Float64)
    #         return x + y * z
    #     end

    #     foo(x) = x
    #     foo(x::Real) = 5x
    #     foo(x::Real, y) = 5x + 10y
    #     foo(g::Float64, d::Real) = 5g * d

    # end))
end
