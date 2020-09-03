@testset "code_transformation/util" begin

    # Test Nabla.unionise_type.
    @test Nabla.unionise_type(:Real) == :(Union{Real, Node{<:Real}})
    @test Nabla.unionise_type(:(T{V})) == :(Union{T{V}, Node{<:T{V}}})
    @test Nabla.unionise_type(:(Vararg)) == :(Vararg{Union{Any, Node{<:Any}}})
    @test Nabla.unionise_type(:(Vararg{T})) == :(Vararg{Union{T, Node{<:T}}})
    @test Nabla.unionise_type(:(T{V} where V)) ==
        :(Union{T{V} where V, Node{<:(T{V} where V)}})
    @test Nabla.unionise_type(:(Vararg{T} where T)) ==
        :(Vararg{Union{T where T, Node{<:(T where T)}}})
    @test Nabla.unionise_type(:(Vararg{<:T})) == :(Vararg{Union{<:T, Node{<:T}}})

    # Test Nabla.replace_body.
    @test Nabla.replace_body(:Real, :Float64) == :Float64
    @test Nabla.replace_body(:Real, :(Union{Float64, foo})) == :(Union{Float64, foo})
    @test Nabla.replace_body(:(T where T), :(Node{T})) == :(Node{T} where T)
    @test Nabla.replace_body(:(T where T <: A{V} where V), :(Node{T})) ==
        :(Node{T} where T <: A{V} where V)

    # Test Nabla.get_body.
    @test Nabla.get_body(:Real) == :Real
    @test Nabla.get_body(:(Node{T})) == :(Node{T})
    @test Nabla.get_body(:(Node{T} where T)) == :(Node{T})
    @test Nabla.get_body(:(Node{T} where T<:A{Q} where Q)) == :(Node{T})
    @test Nabla.get_body(:(Node{A{T} where T})) == :(Node{A{T} where T})
    @test Nabla.get_body(:(Node{A{T} where T} where A)) == :(Node{A{T} where T})

    # Test Nabla.get_types.
    @test Nabla.get_types(:(Tuple{T})) == [:T]
    @test Nabla.get_types(:(Tuple{T, T})) == [:T, :T]
    @test Nabla.get_types(:(Tuple{T{V}})) == [:(T{V})]
    @test Nabla.get_types(:(Tuple{T{<:V}})) == [:(T{<:V})]

    # Test Nabla.isa_vararg.
    @test Nabla.isa_vararg(:foo) == false
    @test Nabla.isa_vararg(:Vararg) == true
    @test Nabla.isa_vararg(:(Node{T} where T)) == false
    @test Nabla.isa_vararg(:(Vararg{T} where T)) == true
    @test Nabla.isa_vararg(:(Vararg{T, N} where T where N)) == true

    # Test Nabla.remove_vararg.
    @test Nabla.remove_vararg(:Vararg) == (:Any, :Vararg)
    @test Nabla.remove_vararg(:(Node{T})) == (:(Node{T}), :nothing)
    @test Nabla.remove_vararg.([:Vararg]) == [(:Any, :Vararg)]
    @test Nabla.remove_vararg.(:Vararg) == (:Any, :Vararg)
    @test Nabla.remove_vararg.([:Vararg, :Vararg]) == [(:Any, :Vararg), (:Any, :Vararg)]
    @test Nabla.remove_vararg(:Real) == (:Real, :nothing)
    @test Nabla.remove_vararg(:(Vararg{T} where T)) == (:(T where T), :Vararg)
    @test Nabla.remove_vararg(:(Vararg{T, N} where T<:Real)) == (:(T where T<:Real), :N)

    # Test Nabla.replace_vararg.
    @test Nabla.replace_vararg(:(U{T, N{T}}), (:V, :nothing)) == :(U{T, N{T}})
    @test Nabla.replace_vararg(:(Real), (:Any, :Vararg)) == :(Vararg{Real})
    @test Nabla.replace_vararg(:(T where T), (:T, :Vararg)) == :(Vararg{T} where T)
    @test Nabla.replace_vararg(:(T where T), (:T, :2)) == :(Vararg{T, 2} where T)
    @test Nabla.replace_vararg(:(U{T} where T), (:T, :N)) == :(Vararg{U{T}, N} where T)
    @test Nabla.replace_vararg(:(U{T} where N), (:T, :N)) == :(Vararg{U{T}, N} where N)

    @testset "parse_kwargs" begin
        @test Nabla.parse_kwargs(:(())) == NamedTuple()
        @test Nabla.parse_kwargs(:(NamedTuple())) == NamedTuple()
        @test Nabla.parse_kwargs(:((a = 1, b = 2))) == NamedTuple{(:a, :b)}((1, 2))
        @test Nabla.parse_kwargs(:((; a = 1, b = 2))) == NamedTuple{(:a, :b)}((1, 2))
        @test Nabla.parse_kwargs(:((a = sum(1:10), b = :c))) ==
            NamedTuple{(:a, :b)}((:(sum(1:10)), :(:c)))
        @test Nabla.parse_kwargs(:((; a = sum(1:10), b = :c))) ==
            NamedTuple{(:a, :b)}((:(sum(1:10)), :(:c)))
        @test_throws ArgumentError Nabla.parse_kwargs(:([a => 2]))
    end

    @testset "parse_is_node" begin
        @test Nabla.parse_is_node(:([])) == Bool[]
        @test Nabla.parse_is_node(:([true])) == [true]
        @test Nabla.parse_is_node(:([true, false])) == [true, false]
        @test_throws ArgumentError Nabla.parse_is_node(:((true, false)))
    end

    @testset "node_type" begin
        @test Nabla.node_type(:(Vararg{Int64, N} where N)) == :(Vararg{Node{<:Int64}})
        @test Nabla.node_type(:Float32) == :(Node{<:Float32})
    end
end
