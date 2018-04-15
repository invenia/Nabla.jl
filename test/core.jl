@testset "core" begin

let

    # test Forward construction and usage.
    let
        f′ = Forward(abs2)
        @test f′(5.0) == abs2(5.0)
        @show f′
        @show f′.tape[1]
        @show f′.tape[2]
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
