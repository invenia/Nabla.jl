# A toy sensitivity which is clearly incorrect.
core_foo(x) = x
@∇primitive core_foo
Nabla.has∇definition(::typeof(core_foo), ::Real) = true
Nabla.∇(::typeof(core_foo), ::Type{Val{1}}, _, y, ȳ, x) = 2 * ȳ

# A varargs sensitivity which is also clearly incorrect.
core_bar(x...) = sum(x)
@∇primitive core_bar
Nabla.has∇definition(::typeof(core_bar), ::Vararg{Real}) = true
Nabla.∇(::typeof(core_bar), ::Type{Val{N}}, _, y, ȳ, x...) where N = N

@testset "core" begin

# let
#     # Test Forward construction and usage.
#     let
#         f, args = x->abs2(abs2(x)), (5.0,)
#         tape = forward(f, args...)
#         display(tape)
#         println()
#         p = preprocess(forward, tape, init_rvs_tape(tape), f, args...)
#         display(p)
#         println()

#         f, args = x->abs2(x * 5), (2.5,)
#         tape = forward(f, args...)
#         display(tape)
#         println()
#         p = preprocess(forward, tape, init_rvs_tape(tape), f, args...)
#         display(p)
#         println()

#         @show ∇(x->abs2(abs2(x)))(5.0)
#         @show ∇(x->abs2(x * 5))(2.5)
#     end
# end

# # Check that the convenience implementation of ∇ works as intended.
# let
#     f(x, y) = 2x + y
#     ∇f = ∇(f)
#     ∇f_out = ∇(f; get_output=true)

#     @test_throws ErrorException ∇f(randn(5), randn(5))
#     x, y = randn(), randn()
#     ∇z = ∇(overdub(∇Ctx, f)(Leaf.(Ref(Tape()), (x, y))...))
#     @test ∇f(x, y) == (∇z[1], ∇z[2])
#     z, (∇x, ∇y) = ∇f_out(x, y)
#     @test z.val == f(x, y)
#     @test (∇x, ∇y) == ∇f(x, y)
# end

    # Test that intercepts work as expected for finite-arg and varargs using toy primitives.
    let
        x1, x2, x3, ȳ = 5.0, 4.0, 3.1, one(Float64)
        @test ∇(core_foo)(x1)[1] == ∇(core_foo, Val{1}, nothing, core_foo(x1), ȳ, x1)
        @test ∇(core_bar)(x1)[1] == ∇(core_bar, Val{1}, nothing, core_bar(x1), ȳ, x1)
        y = forward(core_bar, x1, x2)
        p = preprocess(forward, y, init_rvs_tape(y), core_bar, x1, x2)
        @test ∇(core_bar)(x1, x2)[1] == ∇(core_bar, Val{1}, nothing, core_bar(x1), ȳ, x1, x2)
        @test ∇(core_bar)(x1, x2)[2] == ∇(core_bar, Val{2}, nothing, core_bar(x1), ȳ, x1, x2)
    end

# # Tests for zero'd and one'd containers.
# let
#     import Nabla: zerod_container, oned_container
#     @test zerod_container(1.0) == 0.0
#     @test zerod_container(1) == 0
#     @test oned_container(1.0) == 1.0
#     @test oned_container(5) == 1

#     @test zerod_container(randn(5)) == zeros(5)
#     @test oned_container(randn(5)) == ones(5)
#     @test zerod_container(randn(5, 4, 3, 2, 1)) == zeros(5, 4, 3, 2, 1)
#     @test oned_container(randn(5, 4, 3, 2, 1)) == ones(5, 4, 3, 2, 1)

#     @test zerod_container((1.0, 1)) == (0.0, 0)
#     @test oned_container((0, 0.0)) == (1, 1.0)
#     @test zerod_container((randn(), randn(5))) == (0.0, zeros(5))
#     @test oned_container((randn(5), randn())) == (ones(5), 1.0)
#     @test zerod_container((1.0, (randn(5), randn(5)))) == (0.0, (zeros(5), zeros(5)))
#     @test oned_container((randn(), (randn(5), randn(5)))) == (1.0, (ones(5), ones(5)))

#     @test zerod_container([[1.0], [1.0]]) == [[0.0], [0.0]]
#     @test oned_container([[5.0], [4.0]]) == [[1.0], [1.0]]

#     @test zerod_container(Dict("a"=>5.0, "b"=>randn(3))) == Dict("a"=>0.0, "b"=>zeros(3))
#     @test oned_container(Dict("a"=>5.0, "b"=>randn(3))) == Dict("a"=>1.0, "b"=>ones(3))
# end

end # testset "core"
