@testset "Indexing" begin
    @testset "Int" begin
        leaf = Leaf(Tape(), 5 * [1, 1, 1, 1, 1])
        y = getindex(leaf, 1)
        @test unbox(y) == 5
        @test ∇(y, one(unbox(y)))[leaf] == [1, 0, 0, 0, 0]
    end

    @testset "Vector" begin
        x = Leaf(Tape(), 10 * [1, 1, 1])
        y = x[2:3]
        @test unbox(y) == [10, 10]
        @test ∇(y, oneslike(unbox(y)))[x] == [0, 1, 1]
    end

    @testset "Overlapping indices (#139)" begin
        x = Leaf(Tape(), 10 * [1, 1, 1])
        y = x[[2, 3, 3]]
        @test unbox(y) == [10, 10, 10]
        @test ∇(y, oneslike(unbox(y)))[x] == [0, 1, 2]
    end
end
