# Extend mean naively for the purposes of testing
function Statistics.mean(x::Node{<:AbstractArray}; dims=:)
    if dims === Colon()
        mapfoldl(identity, Base.add_sum, x) / length(x)
    else
        mapreduce(identity, Base.add_sum, x; dims=dims) / size(x, dims)
    end
end

@testset "Reduce dim" begin
    let rng = MersenneTwister(123456)
        # mapreducedim on a single-dimensional array should be consistent with mapreduce.
        x = Leaf(Tape(), [1.0, 2.0, 3.0, 4.0, 5.0])
        s = 5.0 * mapreduce(abs2, +, x, dims=1)
        @test ∇(s, oneslike(s.val))[x] ≈ 5.0 * [2.0, 4.0, 6.0, 8.0, 10.0]

        # mapreduce on a two-dimensional array when reduced over a single dimension
        # should give different results to mapreduce over the same array.
        x2_ = reshape([1.0, 2.0, 3.0, 4.0,], (2, 2))
        x2 = Leaf(Tape(), x2_)
        s = mapreduce(abs2, +, x2, dims=1)
        @test ∇(s, oneslike(s.val))[x2] ≈ 2.0 * x2_

        # mapreducedim under `exp` should trigger the first conditional in the ∇ impl.
        x3_ = randn(rng, 5, 4)
        x3 = Leaf(Tape(), x3_)
        s = mapreduce(exp, +, x3, dims=1)
        @test ∇(s, oneslike(s.val))[x3] == exp.(x3_)

        # mapreducedim under an anonymous-function should trigger fmad.
        x4_ = randn(rng, 5, 4)
        x4 = Leaf(Tape(), x4_)
        s = mapreduce(x->x*x, +, x4, dims=2)
        @test ∇(s, oneslike(s.val))[x4] == 2x4_

        # Check that `sum` works correctly with `Node`s.
        x_sum = Leaf(Tape(), randn(rng, 5, 4, 3))
        @test sum(x_sum, dims=[2, 3]).val == mapreduce(identity, +, x_sum, dims=[2, 3]).val

        # Ensure that the underlying value is correct in the presence of keyword arguments
        x5_ = ones(5, 4, 3)
        x5 = Leaf(Tape(), x5_)
        @test sum(x5, dims=1).val == sum(x5_, dims=1) == fill(5.0, (1, 4, 3))
        @test sum(x5, dims=2).val == sum(x5_, dims=2) == fill(4.0, (5, 1, 3))
        @test sum(x5, dims=3).val == sum(x5_, dims=3) == fill(3.0, (5, 4, 1))
        @test sum(x5).val == 60.0
        @test sum(x5).f === Nabla._mapreduce
        @test mean(x5, dims=1).val == mean(x5_, dims=1) == fill(1.0, (1, 4, 3))
        @test mean(x5, dims=2).val == mean(x5_, dims=2) == fill(1.0, (5, 1, 3))
        @test mean(x5, dims=3).val == mean(x5_, dims=3) == fill(1.0, (5, 4, 1))
        @test mean(x5).val == 1.0
        @test mean(x5).f === Base.:/

        # Issue #123
        x6_ = collect(1:10)
        tens = (fill(10.0, (10,)), fill(10.0, (10, 1)))
        @test ∇(x->sum(sum(x, dims=2)))(x6_) == (oneslike(x6_),)
        @test ∇((x, y)->sum(sum(x, dims=2) .+ sum(y, dims=2)'))(x6_, x6_) == tens
        @test ∇((x, y)->sum(x .+ y'))(x6_, x6_) == tens
    end
end
