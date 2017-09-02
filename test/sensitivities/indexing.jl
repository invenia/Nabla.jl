@testset "Indexing" begin

let
    leaf = Leaf(Tape(), 5 * [1, 1, 1, 1, 1])
    y = getindex(leaf, 1)
    @test y.val == 5
    @test ∇(y, one(y.val))[leaf] == [1, 0, 0, 0, 0]
end

let
    x = Leaf(Tape(), 10 * [1, 1, 1])
    y = x[2:3]
    @test y.val == [10, 10]
    @test ∇(y, ones(y.val))[x] == [0, 1, 1]
end

end
# Tests for `view`. Not currently in use because `view` is actually a bit awkward.
# @test isdefined(:view)
# @test method_exists(DiffBase.view, Tuple{Vararg})
# @test method_exists(DiffCore.get_original, Tuple{typeof(view)})
# @test DiffCore.get_original(view) == Base.view
# @test method_exists(∇, Tuple{typeof(view), Type{Arg{1}}, Any, Any, Any, Any, Vararg})
# @test method_exists(∇, Tuple{Any, typeof(view), Type{Arg{1}}, Any, Any, Any, Any, Vararg})

# let
#     leaf = Leaf(Tape(), 5 * [1, 1, 1, 1, 1])
#     y = view(leaf, 1)
#     println(typeof(y.val[1]))
#     println(typeof(5))
#     println(∇(y)[leaf])
#     @test y.val[1] == 5
#     @test ∇(y)[leaf] == [1, 0, 0, 0, 0]
# end

# let
#     x = Leaf(Tape(), 10 * [1, 1, 1])
#     y = view(x, 2:3)
#     @test y.val[1:2] == [10, 10]
#     @test ∇(y)[x] == [0, 1, 1]
# end
