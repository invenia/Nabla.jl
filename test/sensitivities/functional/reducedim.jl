@testset "functional/mapreducedim" begin

    # mapreducedim on a single-dimensional array should be consistent with mapreduce.
    x = Leaf(Tape(), [1.0, 2.0, 3.0, 4.0, 5.0])
    s = 5.0 * mapreducedim(abs2, +, x, 1)[1]
    @test ∇(s)[x] ≈ 5.0 * [2.0, 4.0, 6.0, 8.0, 10.0]

    # mapreducedim on a two-dimensional array when reduced over a single dimension should
    # give different results to mapreduce over the same array.
    x2_ = reshape([1.0, 2.0, 3.0, 4.0,], (2, 2))
    x2 = Leaf(Tape(), x2_)
    s = mapreducedim(abs2, +, x2, 1)
    @test ∇(s, ones(s.val))[x] ≈ 2.0 * x2_
end
