@testset "core" begin

let
    # Test Forward construction and usage.
    let
        f, args = x->abs2(abs2(x)), (5.0,)
        tape = forward(f, args...)
        display(tape)
        println()
        p = preprocess(forward, tape, init_rvs_tape(tape), f, args...)
        display(p)
        println()

        f, args = x->abs2(x * 5), (2.5,)
        tape = forward(f, args...)
        display(tape)
        println()
        p = preprocess(forward, tape, init_rvs_tape(tape), f, args...)
        display(p)
        println()

        @show ∇(x->abs2(abs2(x)))(5.0)
        @show ∇(x->abs2(x * 5))(2.5)
    end
end

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
