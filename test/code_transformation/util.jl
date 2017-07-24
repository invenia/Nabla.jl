@testset "code_transformation/util" begin

    # Test DiffCore.unionise_type.
    @test DiffCore.unionise_type(:Real) == :(Union{Real, Node{<:Real}})
    @test DiffCore.unionise_type(:(T{V})) == :(Union{T{V}, Node{<:T{V}}})
    @test DiffCore.unionise_type(:(Vararg)) == :(Vararg{Union{Any, Node{<:Any}}})
    @test DiffCore.unionise_type(:(Vararg{T})) == :(Vararg{Union{T, Node{<:T}}})
    @test DiffCore.unionise_type(:(T{V} where V)) ==
        :(Union{T{V} where V, Node{<:(T{V} where V)}})
    @test DiffCore.unionise_type(:(Vararg{T} where T)) ==
        :(Vararg{Union{T where T, Node{<:(T where T)}}})
    @test DiffCore.unionise_type(:(Vararg{<:T})) == :(Vararg{Union{<:T, Node{<:T}}})

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

    # Test DiffCore.get_types.
    @test DiffCore.get_types(:(Tuple{T})) == [:T]
    @test DiffCore.get_types(:(Tuple{T, T})) == [:T, :T]
    @test DiffCore.get_types(:(Tuple{T{V}})) == [:(T{V})]
    @test DiffCore.get_types(:(Tuple{T{<:V}})) == [:(T{<:V})]

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
