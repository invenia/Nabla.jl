# @testset "Reduce dim" begin
#     let rng = MersenneTwister(123456)
#         # mapreducedim on a single-dimensional array should be consistent with mapreduce.
#         x = Leaf(Tape(), [1.0, 2.0, 3.0, 4.0, 5.0])
#         s = overdub(∇Ctx, x->5.0 * mapreducedim(abs2, +, x, 1)[1])(x)
#         @test ∇(s)[x] ≈ 5.0 * [2.0, 4.0, 6.0, 8.0, 10.0]

#         # mapreducedim on a two-dimensional array when reduced over a single dimension
#         # should give different results to mapreduce over the same array.
#         x2_ = reshape([1.0, 2.0, 3.0, 4.0,], (2, 2))
#         x2 = Leaf(Tape(), x2_)
#         s = overdub(∇Ctx, x->mapreducedim(abs2, +, x, 1))(x2)
#         @test ∇(s, ones(s.val))[x2] ≈ 2.0 * x2_

#         # mapreducedim under `exp` should trigger the first conditional in the ∇ impl.
#         x3_ = randn(rng, 5, 4)
#         x3 = Leaf(Tape(), x3_)
#         s = overdub(∇Ctx, x->mapreducedim(exp, +, x, 1))(x3)
#         @test ∇(s, ones(s.val))[x3] == exp.(x3_)

#         # mapreducedim under an anonymous-function should trigger fmad.
#         x4_ = randn(rng, 5, 4)
#         x4 = Leaf(Tape(), x4_)
#         s = overdub(∇Ctx, x_->mapreducedim(x->x*x, +, x_, 2))(x4)
#         @test ∇(s, ones(s.val))[x4] == 2x4_

#         # Check that `sum` works correctly with `Node`s.
#         x_sum = overdub(∇Ctx, x->Leaf(Tape(), x))(randn(rng, 5, 4, 3))
#         s = mapreducedim(identity, +, x_sum; dims=[2, 3])
#         @test sum(x_sum, [2, 3]).val == mapreducedim(identity, +, x_sum, [2, 3]).val
#     end
# end
