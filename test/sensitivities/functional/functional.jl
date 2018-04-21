using SpecialFunctions
using DiffRules: diffrule, hasdiffrule
using Nabla: fmad

struct Uniform{T<:Real}
    a::T
    b::T
end
Base.rand(rng::AbstractRNG, u::Uniform, N::Int) = rand(rng, N) .* (u.b - u.a) .+ u.a
Base.rand(rng::AbstractRNG, u::Uniform) = rand(rng) * (u.b - u.a) + u.a

@testset "Functional" begin

    let
        rng = MersenneTwister(123456)

        # Check that `broadcastsum!` works as intended.
        let
            Z, X, Y = [1 2; 3 4], [1 2], [5 6; 7 8]
            @test Nabla.broadcastsum!(x->2x, false, copy(X), Z) == [2 + 6 4 + 8]
            @test Nabla.broadcastsum!(x->2x, true, copy(X), Z) == X + [2 + 6 4 + 8]
            @test Nabla.broadcastsum!(x->2x, false, copy(Y), Z) == 2Z
            @test Nabla.broadcastsum!(x->2x, true, copy(Y), Z) == Y + 2Z
        end

        # Check that `broadcast` returns the correct gradient under the defined unary functions.
        function check_unary_broadcast(f, x)
            # x_ = Leaf(Tape(), x)
            # s = overdub(∇Ctx, x->broadcast(f, x))(x_)
            # return ∇(s, fill(1, size(s.val)))[x_] ≈ ∇.(f, Val{1}, x)
            ȳ, v = randn(rng, size(x)), randn(rng, size(x))
            @test check_unary(x->broadcast(f, x), ȳ, x, v)
        end
        # for (package, f) in Nabla.unary_sensitivities
        #     domain = domain1(eval(f))
        #     isnull(domain) && error("Could not determine domain for $f.")
        #     x_dist = Uniform(get(domain)...)
        #     x = rand(rng, x_dist, 100)
        #     @test check_unary_broadcast(eval(f), x)
        # end

    #     # Check that `broadcast` returns the correct gradient under each implemented binary
    #     # function.
    #     function check_binary_broadcast(f, x, y)
    #         tape = Tape()
    #         x_, y_ = Leaf(tape, x), Leaf(tape, y)
    #         s = overdub(∇Ctx, (x_, y_)->broadcast(f, x_, y_))(x_, y_)
    #         ∇s = ∇(s, fill(1, size(s.val)))
    #         ∇x = broadcast((z, z̄, x, y)->∇(f, Val{1}, nothing, z, z̄, x, y),
    #                        s.val, fill(1, size(s.val)), x, y)
    #         ∇y = broadcast((z, z̄, x, y)->∇(f, Val{2}, nothing, z, z̄, x, y),
    #                        s.val, fill(1, size(s.val)), x, y)
    #         @test broadcast(f, x, y) == s.val
    #         @test ∇s[x_] ≈ ∇x
    #         @test ∇s[y_] ≈ ∇y
    #     end
    #     function check_binary_broadcast(f, x::Real, y)
    #         tape = Tape()
    #         x_, y_ = Leaf(tape, x), Leaf(tape, y)
    #         s = overdub(∇Ctx, (x_, y_)->broadcast(f, x_, y_))(x_, y_)
    #         ∇s = ∇(s, fill(1, size(s.val)))
    #         ∇x = sum(broadcast((z, z̄, x, y)->∇(f, Val{1}, nothing, z, z̄, x, y),
    #                            s.val, fill(1, size(s.val)), x, y))
    #         ∇y = broadcast((z, z̄, x, y)->∇(f, Val{2}, nothing, z, z̄, x, y),
    #                        s.val, fill(1, size(s.val)), x, y)
    #         @test broadcast(f, x, y) == s.val
    #         @test ∇s[x_] ≈ ∇x
    #         @test ∇s[y_] ≈ ∇y
    #     end
    #     function check_binary_broadcast(f, x, y::Real)
    #         tape = Tape()
    #         x_, y_ = Leaf(tape, x), Leaf(tape, y)
    #         s = overdub(∇Ctx, (x_, y_)->broadcast(f, x_, y_))(x_, y_)
    #         ∇s = ∇(s, fill(1, size(s.val)))
    #         ∇x = broadcast((z, z̄, x, y)->∇(f, Val{1}, nothing, z, z̄, x, y),
    #                        s.val, fill(1, size(s.val)), x, y)
    #         ∇y = sum(broadcast((z, z̄, x, y)->∇(f, Val{2}, nothing, z, z̄, x, y),
    #                            s.val, fill(1, size(s.val)), x, y))
    #         @test broadcast(f, x, y) == s.val
    #         @test ∇s[x_] ≈ ∇x
    #         @test ∇s[y_] ≈ ∇y
    #     end
    #     # for (package, f) in Nabla.binary_sensitivities

    #     #     # TODO: More care needs to be taken to test the following.
    #     #     if hasdiffrule(package, f, 2)
    #     #         ∂f∂x, ∂f∂y = diffrule(package, f, :x, :y)
    #     #     else
    #     #         ∂f∂x, ∂f∂y = :∂f∂x, :∂f∂y
    #     #     end

    #     #     # TODO: Implement the edge cases for functions differentiable in only either
    #     #     # argument.
    #     #     (∂f∂x == :NaN || ∂f∂y == :NaN) && continue
    #     #     domain = domain2(eval(f))
    #     #     isnull(domain) && error("Could not determine domain for $f.")
    #     #     (x_lb, x_ub), (y_lb, y_ub) = get(domain)
    #     #     x_dist, y_dist = Uniform(x_lb, x_ub), Uniform(y_lb, y_ub)
    #     #     x, y = rand(rng, x_dist, 100), rand(rng, y_dist, 100)
    #     #     check_binary_broadcast(eval(f), x, y)
    #     #     check_binary_broadcast(eval(f), rand(rng, x_dist), y)
    #     #     check_binary_broadcast(eval(f), x, rand(rng, y_dist))
    #     # end
        
    #     let # Ternary functions (because it's useful to check I guess.)
    #         f = (x, y, z)->x * y + y * z + x * z
    #         x, y, z = randn(rng, 5), randn(rng, 5), randn(rng, 5)
    #         x_, y_, z_ = Leaf.(Ref(Tape()), (x, y, z))
    #         s_ = overdub(∇Ctx,(x_, y_, z_)->broadcast(f, x_, y_, z_))(x_, y_, z_)
    #         ∇s = ∇(s_, fill(1, size(s_.val)))
    #         @test s_.val == broadcast(f, x, y, z)
    #         @test ∇s[x_] == getindex.(broadcast((x, y, z)->fmad(f, (x, y, z)), x, y, z), 1)
    #         @test ∇s[y_] == getindex.(broadcast((x, y, z)->fmad(f, (x, y, z)), x, y, z), 2)
    #         @test ∇s[z_] == getindex.(broadcast((x, y, z)->fmad(f, (x, y, z)), x, y, z), 3)
    #     end

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
