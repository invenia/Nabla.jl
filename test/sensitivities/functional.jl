using SpecialFunctions
using DiffRules: diffrule, hasdiffrule
using Nabla: fmad
import Base: rand
import Base.Broadcast: broadcast_shape

struct Uniform{T<:Real}
    a::T
    b::T
end
rand(rng::AbstractRNG, u::Uniform, N::Int...) = rand(rng, N...) .* (u.b - u.a) .+ u.a

@testset "Functional" begin

    let
        rng, N, N′ = MersenneTwister(123456), 5, 6

        # Check that `broadcastsum!` works as intended.
        let
            Z, X, Y = [1 2; 3 4], [1 2], [5 6; 7 8]
            @test Nabla.broadcastsum!(x->2x, false, copy(X), Z) == [2 + 6 4 + 8]
            @test Nabla.broadcastsum!(x->2x, true, copy(X), Z) == X + [2 + 6 4 + 8]
            @test Nabla.broadcastsum!(x->2x, false, copy(Y), Z) == 2Z
            @test Nabla.broadcastsum!(x->2x, true, copy(Y), Z) == Y + 2Z
        end

        # Check arbitrary-arity broadcasts.
        function check_broadcast(f, x...)
            v = map(x->randn(rng, size(x)...), x)
            z̄ = randn(rng, broadcast_shape(map(size, x)...)...)
            @test check_errs((x...)->broadcast(f, x...), z̄, x, v)
        end

        # Check arbitrary-arity maps.
        function check_map(f, x...)
            z̄, v = randn(rng, size(x[1])...), map(x->randn(rng, size(x)...), x)
            @test check_errs((x...)->map(f, x...), z̄, x, v)
        end

        # # Test that all of the unary DiffRules functions broadcast properly.
        # for (package, f, f) in Nabla.unary_sensitivities
        #     domain = domain1(f)
        #     isnull(domain) && error("Could not determine domain for $f.")
        #     x_dist = Uniform(get(domain)...)
        #     check_broadcast(f, rand(rng, x_dist))
        #     check_broadcast(f, rand(rng, x_dist, N))
        #     check_broadcast(f, rand(rng, x_dist, N, N′))
        # end

        # # Test all of the binary DiffRules functions broadcast properly.
        # for (package, f_sym, f) in Nabla.binary_sensitivities
        #     ∂f∂x, ∂f∂y = diffrule(package, f_sym, :x, :y)
        #     if ∂f∂x != :NaN && ∂f∂y != :NaN
        #         domain = domain2(f)
        #         isnull(domain) && error("Could not determine domain for $f.")
        #         (x_lb, x_ub), (y_lb, y_ub) = get(domain)
        #         x_dist, y_dist = Uniform(x_lb, x_ub), Uniform(y_lb, y_ub)
        #         check_broadcast(f, rand(rng, x_dist, N), rand(rng, y_dist, N))
        #         check_broadcast(f, rand(rng, x_dist), rand(rng, y_dist, N))
        #         check_broadcast(f, rand(rng, x_dist, N), rand(rng, y_dist))
        #         check_broadcast(f, rand(rng, x_dist, N, N′), rand(rng, y_dist, N))
        #         check_broadcast(f, rand(rng, x_dist, N, N′), rand(rng, y_dist))
        #     end
        # end

        # # Test a unary composite function.
        # let
        #     f = x->cos(exp(x))
        #     check_broadcast(f, randn(rng))
        #     check_broadcast(f, randn(rng, N))
        #     check_broadcast(f, randn(rng, N, N′))
        #     # check_map(f, randn(rng))
        #     check_map(f, randn(rng, N))
        #     check_map(f, randn(rng, N, N′))
        # end

        # # Test a binary composite function.
        # let
        #     f = (x, y)->x * exp(y) + x
        #     check_broadcast(f, rand(rng), randn(rng))
        #     check_broadcast(f, randn(rng, N), randn(rng))
        #     check_broadcast(f, randn(rng, N), randn(rng, N))
        #     check_broadcast(f, randn(rng, N, N′), randn(rng, N))
        #     check_broadcast(f, randn(rng, N, N′), randn(rng, N, N′))
        #     # check_map(f, rand(rng), randn(rng))
        #     check_map(f, randn(rng, N), randn(rng, N))
        #     check_map(f, randn(rng, N, N′), randn(rng, N, N′))
        # end

        # # Test some arbitrary ternary composite function.
        # let 
        #     f = (x, y, z)->x * exp(y) + y * cos(z) + sin(x * z)
        #     check_broadcast(f, randn(rng), randn(rng), randn(rng))
        #     check_broadcast(f, randn(rng), randn(rng, N), randn(rng))
        #     check_broadcast(f, randn(rng, N), randn(rng, N), randn(rng, N))
        #     check_broadcast(f, randn(rng, N), randn(rng, N, N′), randn(rng))
        #     check_broadcast(f, randn(rng, N, N′), randn(rng, N, N′), randn(rng, N, N′))
        #     # check_map(f, randn(rng), randn(rng), randn(rng))
        #     check_map(f, randn(rng, N), randn(rng, N), randn(rng, N))
        #     check_map(f, randn(rng, N, N′), randn(rng, N, N′), randn(rng, N, N′))
        # end

        let
            f = x->cos(sin(x))
            @test check_errs(x->mapreduce(f, +, x), randn(rng), randn(rng, N), randn(rng, N))
            @test check_errs(x->mapreduce(f, +, x), randn(rng), randn(rng, N, N′), randn(rng, N, N′))
            @test check_errs(x->sum(f, x), randn(rng), randn(rng, N), randn(rng, N))
            @test check_errs(sum, randn(rng), randn(rng, N), randn(rng, N))
            @test check_errs(x->mapreducedim(f, +, x, 2), randn(rng, N, 1), randn(rng, N, N′), randn(rng, N, N′))
            @test check_errs(x->mapreducedim(f, +, x, 1), randn(rng, 1, N′), randn(rng, N, N′), randn(rng, N, N′))
        end

    # # # Check that the number of allocations which happen in the reverse pass of `map` and
    # # # `broadcast` is invariant of the size of the data structure of which we are
    # # # mapping / broadcasting.
    # # for f in [:map, :broadcast]
    # #     let
    # #         @eval foo_small() = sum($f(tanh, Leaf(Tape(), randn(10, 10))))
    # #         @eval foo_large() = sum($f(tanh, Leaf(Tape(), randn(10, 100))))
    # #         @test allocs(@benchmark foo_small()) == allocs(@benchmark foo_large())
    # #         @test allocs(@benchmark ∇(foo_small())) == allocs(@benchmark ∇(foo_large()))
    # #     end
    # # end
    end
end


# @testset "Reduce" begin
#     let rng = MersenneTwister(123456)
#         import Nabla.fmad

#         # Check that `mapreduce`, `mapfoldl`and `mapfoldr` work as expected with all unary
#         # functions, some composite functions which use FMAD under both `+` and `*`.
#         let N = 3
#             for functional in (mapreduce, mapfoldl, mapfoldr)

#                 # Sensitivities implemented in Base.
#                 for (package, f) in Nabla.unary_sensitivities

#                     # Generate some data and get the function to be mapped.
#                     f = eval(f)
#                     domain = domain1(f)
#                     isnull(domain) && error("Could not determine domain for $f.")
#                     lb, ub = get(domain)
#                     x = rand(rng, N) .* (ub - lb) .+ lb

#                     # Test +.
#                     x_ = Leaf(Tape(), x)
#                     s = overdub(∇Ctx, x_->functional(f, +, x_))(x_)
#                     @test ∇(s)[x_] ≈ ∇.(f, Val{1}, x)
#                 end

#                 # Some composite sensitivities.
#                 composite_functions = (x->5x, x->1 / (1 + x), x->10+x)
#                 for f in composite_functions

#                     # Generate some data.
#                     x = randn(rng, 100)

#                     # Test +.
#                     x_ = Leaf(Tape(), x)
#                     s = overdub(∇Ctx, x->functional(f, +, x))(x_)
#                     @test s.val == functional(f, +, x)
#                     @test ∇(s)[x_] ≈ map(x->fmad(f, (x,), Val{1}), x)
#                 end
#             end
#         end

#         # Check that `reduce`, `foldl` and `foldr` work as expected for `+` and `*`.
#         let
#             for functional in (reduce, foldl, foldr)

#                 # Test `+`.
#                 x = randn(rng, 100)
#                 x_ = Leaf(Tape(), x)
#                 s = overdub(∇Ctx, x->functional(+, x))(x_)
#                 @test s.val == functional(+, x)
#                 @test ∇(s)[x_] ≈ ones(x)
#             end
#         end

#         # Check that `sum` and `prod` work as expected under all implemented unary functions
#         # and some composite functions which use FMAD.
#         let N = 5
#             # Sensitivities implemented in Base.
#             for (package, f) in Nabla.unary_sensitivities

#                 # Generate some data and get the function to be mapped.
#                 f = eval(f)
#                 domain = domain1(f)
#                 isnull(domain) && error("Could not determine domain for $f.")
#                 lb, ub = get(domain)
#                 x = rand(rng, N) .* (ub - lb) .+ lb

#                 # Test +.
#                 x_ = Leaf(Tape(), x)
#                 s = overdub(∇Ctx, x->sum(f, x))(x_)
#                 @test s.val == sum(f, x)
#                 @test ∇(s)[x_] ≈ ∇.(f, Val{1}, x)
#             end

#             # Some composite functions.
#             composite_functions = (x->5x, x->1 / (1 + x), x->10+x)
#             for f in composite_functions

#                 # Generate some data.
#                 x = randn(rng, N)

#                 # Test +.
#                 x_ = Leaf(Tape(), x)
#                 s = overdub(∇Ctx, x->sum(f, x))(x_)
#                 @test s.val == sum(f, x)
#                 @test ∇(s)[x_] ≈ map(x->fmad(f, (x,), Val{1}), x)
#             end
#         end
#     end
# end

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
