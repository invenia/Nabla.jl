using Nabla: Op, forward, Tape, ∇MaybeTagged, ∇Ctx, Arg
import Nabla: ∇, is_atom
using Cassette: istagged

# Create a toy primtiive for testing purposes.
foo_core(x) = sin(x)
∇(::typeof(foo_core), ::Type{Arg{1}}, _, y, ȳ, x) = ȳ * exp(x)
is_atom(ctx::∇Ctx, ::typeof(foo_core), x::∇MaybeTagged{<:∇Scalar}) = istagged(x, ctx)

@testset "core" begin

let
    @test Op(sin, 5).value == sin(5)
    @test Op(sin, 5).f == sin
    @test Op(sin, 5).args == (5,)
    @test Op(sum, ones(10, 10)).value == sum(ones(10, 10))
    @test Op(sum, ones(10, 10); dims=1).value == sum(ones(10, 10); dims=1)
end

let
    @test forward(foo_core, 5) == foo_core(5)
    @test forward(Tape(), foo_core, 5) == foo_core(5)
    @test ∇(foo_core)(5)[1] == ∇(foo_core, Arg{1}, nothing, foo_core(5), 1, 5)
    @test ∇(x->foo_core(x))(5)[1] == ∇(foo_core, Arg{1}, nothing, foo_core(5), 1, 5)
end

# # Check that functions involving `isapprox` can be differentiated
# let
#     f(x) = x ≈ 5.0 ? 1.0 : 3.0 * x
#     g(x) = 5.0 * x
#     h(x) = g(x) ≈ 25.0 ? x : f(x) + g(x)
#     ∇f = ∇(f)
#     ∇h = ∇(h)
#     @test ∇f(5.0) == (0.0,)
#     @test ∇f(6.0) == (3.0,)
#     @test ∇h(5.0) == (1.0,)
#     @test ∇h(6.0) == (8.0,)
#     f(x) = x ≈ [5.0] ? 1.0 : 3.0 * sum(x)
#     ∇f = ∇(f)
#     @test ∇f([5.0]) == ([0.0],)
#     @test ∇f([6.0]) == ([3.0],)
# end

# # Check that functions with extra, unused variables can be differentiated
# let
#     f(a,b,c,d) = a*c
#     ∇f = ∇(f)
#     g(a,b) = 12
#     ∇g = ∇(g)

#     @test ∇f(1,2,3,4) == (3, 0, 1, 0)
#     @test ∇f(1,[2.0],3,4.0) == (3, [0.0], 1, 0.0)
#     @test ∇g(1,2) == (0,0)
# end

# # Check that functions with `zero` and `one` can be differentiated
# let
#     f(a) = zero(a)
#     g(a) = one(a)
#     h(a) = zero(3 * a) + one(4 * a)
#     ∇f = ∇(f)
#     ∇g = ∇(g)
#     ∇h = ∇(h)

#     @test ∇f(1) == (0,)
#     @test ∇f([1]) == ([0],)
#     @test ∇g(4) == (0,)
#     @test ∇h(8) == (0,)
# end

# # Check that the convenience implementation of ∇ works as intended.
# let
#     f(x, y) = 2x + y
#     ∇f = ∇(f)
#     ∇f_out = ∇(f, true)

#     @test_throws MethodError ∇f(randn(5), randn(5))
#     x, y = randn(), randn()
#     ∇z = ∇(f(Leaf.(Tape(), (x, y))...))
#     @test ∇f(x, y) == (∇z[1], ∇z[2])
#     z, (∇x, ∇y) = ∇f_out(x, y)
#     @test z.val == f(x, y)
#     @test (∇x, ∇y) == ∇f(x, y)
# end

# Tests for zero'd and one'd containers.
let
    import Nabla: zerod_container, oned_container
    @test zerod_container(1.0) == 0.0
    @test zerod_container(1) == 0
    @test oned_container(1.0) == 1.0
    @test oned_container(5) == 1

    @test zerod_container(randn(5)) == zeros(5)
    @test oned_container(randn(5)) == ones(5)
    @test zerod_container(randn(5, 4, 3, 2, 1)) == zeros(5, 4, 3, 2, 1)
    @test oned_container(randn(5, 4, 3, 2, 1)) == ones(5, 4, 3, 2, 1)

    @test zerod_container((1.0, 1)) == (0.0, 0)
    @test oned_container((0, 0.0)) == (1, 1.0)
    @test zerod_container((randn(), randn(5))) == (0.0, zeros(5))
    @test oned_container((randn(5), randn())) == (ones(5), 1.0)
    @test zerod_container((1.0, (randn(5), randn(5)))) == (0.0, (zeros(5), zeros(5)))
    @test oned_container((randn(), (randn(5), randn(5)))) == (1.0, (ones(5), ones(5)))

    @test zerod_container([[1.0], [1.0]]) == [[0.0], [0.0]]
    @test oned_container([[5.0], [4.0]]) == [[1.0], [1.0]]

    @test zerod_container(Dict("a"=>5.0, "b"=>randn(3))) == Dict("a"=>0.0, "b"=>zeros(3))
    @test oned_container(Dict("a"=>5.0, "b"=>randn(3))) == Dict("a"=>1.0, "b"=>ones(3))
end

end # testset "core"
