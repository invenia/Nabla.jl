@testset "array" begin
    @test size(Leaf(Tape(), ones(1, 2, 3, 4))) == (1, 2, 3, 4)
    @test length(Leaf(Tape(), 1:3)) == 3
end
