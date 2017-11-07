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
