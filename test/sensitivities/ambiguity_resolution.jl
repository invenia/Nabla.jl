@testset "Ambiguity Resolution" begin
    @testset "fill" begin
        k = fill(Leaf(Tape(), 1.2), 3, 4)
        @test k.pullback !== nothing

        x = fill(Leaf(Tape(), 1.2), (3, 4))
        @test x.pullback !== nothing
    end
end