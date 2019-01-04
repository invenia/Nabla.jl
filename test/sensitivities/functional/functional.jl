using SpecialFunctions
using DiffRules: diffrule, hasdiffrule

@testset "Functional" begin
    # Apparently Distributions.jl doesn't implement the following, so we'll have to do it.
    Random.rand(rng::AbstractRNG, a::Distribution, n::Integer) =
        [rand(rng, a) for _ in 1:n]

    let rng = MersenneTwister(123456)
        import Nabla.fmad

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
            x_ = Leaf(Tape(), x)
            s = broadcast(f, x_)
            return ∇(s, oneslike(s.val))[x_] ≈ ∇.(f, Arg{1}, x)
        end
        for (package, f) in Nabla.unary_sensitivities
            domain = domain1(eval(f))
            domain === nothing && error("Could not determine domain for $f.")
            x_dist = Uniform(domain...)
            x = rand(rng, x_dist, 100)
            @test check_unary_broadcast(eval(f), x)
        end

        # Check that `broadcast` returns the correct gradient under each implemented binary
        # function.
        function check_binary_broadcast(f, x, y)
            tape = Tape()
            x_, y_ = Leaf(tape, x), Leaf(tape, y)
            s = broadcast(f, x_, y_)
            o = oneslike(s.val)
            ∇s = ∇(s, o)
            ∇x = broadcast((z, z̄, x, y)->∇(f, Arg{1}, nothing, z, z̄, x, y),
                           s.val, o, x, y)
            ∇y = broadcast((z, z̄, x, y)->∇(f, Arg{2}, nothing, z, z̄, x, y),
                           s.val, o, x, y)
            @test broadcast(f, x, y) == s.val
            @test ∇s[x_] ≈ ∇x
            @test ∇s[y_] ≈ ∇y
        end
        function check_binary_broadcast(f, x::Real, y)
            tape = Tape()
            x_, y_ = Leaf(tape, x), Leaf(tape, y)
            s = broadcast(f, x_, y_)
            o = oneslike(s.val)
            ∇s = ∇(s, o)
            ∇x = sum(broadcast((z, z̄, x, y)->∇(f, Arg{1}, nothing, z, z̄, x, y),
                               s.val, o, x, y))
            ∇y = broadcast((z, z̄, x, y)->∇(f, Arg{2}, nothing, z, z̄, x, y),
                           s.val, o, x, y)
            @test broadcast(f, x, y) == s.val
            @test ∇s[x_] ≈ ∇x
            @test ∇s[y_] ≈ ∇y
        end
        function check_binary_broadcast(f, x, y::Real)
            tape = Tape()
            x_, y_ = Leaf(tape, x), Leaf(tape, y)
            s = broadcast(f, x_, y_)
            o = oneslike(s.val)
            ∇s = ∇(s, o)
            ∇x = broadcast((z, z̄, x, y)->∇(f, Arg{1}, nothing, z, z̄, x, y),
                           s.val, o, x, y)
            ∇y = sum(broadcast((z, z̄, x, y)->∇(f, Arg{2}, nothing, z, z̄, x, y),
                               s.val, o, x, y))
            @test broadcast(f, x, y) == s.val
            @test ∇s[x_] ≈ ∇x
            @test ∇s[y_] ≈ ∇y
        end
        for (package, f) in Nabla.binary_sensitivities

            # TODO: More care needs to be taken to test the following.
            if hasdiffrule(package, f, 2)
                ∂f∂x, ∂f∂y = diffrule(package, f, :x, :y)
            else
                ∂f∂x, ∂f∂y = :∂f∂x, :∂f∂y
            end

            # TODO: Implement the edge cases for functions differentiable in only either
            # argument.
            (∂f∂x == :NaN || ∂f∂y == :NaN) && continue
            domain = domain2(eval(f))
            domain === nothing && error("Could not determine domain for $f.")
            (x_lb, x_ub), (y_lb, y_ub) = domain
            x_dist, y_dist = Uniform(x_lb, x_ub), Uniform(y_lb, y_ub)
            x, y = rand(rng, x_dist, 100), rand(rng, y_dist, 100)
            check_binary_broadcast(eval(f), x, y)
            check_binary_broadcast(eval(f), rand(rng, x_dist), y)
            check_binary_broadcast(eval(f), x, rand(rng, y_dist))
        end
        #
        let # Ternary functions (because it's useful to check I guess.)
            f = (x, y, z)->x * y + y * z + x * z
            x, y, z = randn(rng, 5), randn(rng, 5), randn(rng, 5)
            x_, y_, z_ = Leaf.(Tape(), (x, y, z))
            s_ = broadcast(f, x_, y_, z_)
            ∇s = ∇(s_, oneslike(s_.val))
            @test s_.val == broadcast(f, x, y, z)
            @test ∇s[x_] == getindex.(broadcast((x, y, z)->fmad(f, (x, y, z)), x, y, z), 1)
            @test ∇s[y_] == getindex.(broadcast((x, y, z)->fmad(f, (x, y, z)), x, y, z), 2)
            @test ∇s[z_] == getindex.(broadcast((x, y, z)->fmad(f, (x, y, z)), x, y, z), 3)
        end

        let
            x, y, tape = 5.0, randn(rng, 5), Tape()
            x_, y_ = Leaf(tape, x), Leaf(tape, y)
            z_ = x_ .+ y_
            z2_ = broadcast(+, x_, y_)
            @test z_.val == x .+ y
            @test ∇(z_, oneslike(z_.val))[x_] == ∇(z2_, oneslike(z2_.val))[x_]
            @test ∇(z_, oneslike(z_.val))[y_] == ∇(z2_, oneslike(z2_.val))[y_]
        end
        let
            x, y, tape = randn(rng, 5), 5.0, Tape()
            x_, y_ = Leaf(tape, x), Leaf(tape, y)
            z_ = x_ * y_
            z2_ = broadcast(*, x_, y_)
            @test z_.val == x .* y
            @test ∇(z_, oneslike(z_.val))[x_] == ∇(z2_, oneslike(z2_.val))[x_]
            @test ∇(z_, oneslike(z_.val))[y_] == ∇(z2_, oneslike(z2_.val))[y_]
        end
        let
            x, y, tape = randn(rng, 5), 5.0, Tape()
            x_, y_ = Leaf(tape, x), Leaf(tape, y)
            z_ = x_ .- y_
            z2_ = broadcast(-, x_, y_)
            @test z_.val == x .- y
            @test ∇(z_, oneslike(z_.val))[x_] == ∇(z2_, oneslike(z2_.val))[x_]
            @test ∇(z_, oneslike(z_.val))[y_] == ∇(z2_, oneslike(z2_.val))[y_]
        end
        let
            x, y, tape = randn(rng, 5), 5.0, Tape()
            x_, y_ = Leaf(tape, x), Leaf(tape, y)
            z_ = x_ / y_
            z2_ = broadcast(/, x_, y_)
            @test z_.val == x ./ y
            @test ∇(z_, oneslike(z_.val))[x_] == ∇(z2_, oneslike(z2_.val))[x_]
            @test ∇(z_, oneslike(z_.val))[y_] == ∇(z2_, oneslike(z2_.val))[y_]
        end
        let
            x, y, tape = 5.0, randn(rng, 5), Tape()
            x_, y_ = Leaf(tape, x), Leaf(tape, y)
            z_ = x_ \ y_
            z2_ = broadcast(\, x_, y_)
            @test z_.val == x .\ y
            @test ∇(z_, oneslike(z_.val))[x_] == ∇(z2_, oneslike(z2_.val))[x_]
            @test ∇(z_, oneslike(z_.val))[y_] == ∇(z2_, oneslike(z2_.val))[y_]
        end

        # Check that dot notation works as expected for all unary function in Nabla for both
        # scalars and arrays.
        function check_unary_dot(f, x)
            x_ = Leaf(Tape(), x)
            z_ = f.(x_)
            z2_ = broadcast(f, x_)
            @test z_.val == f.(x)
            @test ∇(z_, oneslike(z_.val))[x_] == ∇(z2_, oneslike(z2_.val))[x_]
        end
        function check_unary_dot(f, x::∇Scalar)
            x_ = Leaf(Tape(), x)
            z_ = f.(x_)
            @test z_.val == f.(x)
            @test ∇(z_)[x_] == ∇(broadcast(f, x_))[x_]
        end
        for (package, f) in Nabla.unary_sensitivities
            domain = domain1(eval(f))
            domain === nothing && error("Could not determine domain for $f.")
            x_dist = Uniform(domain...)
            check_unary_dot(eval(f), rand(rng, x_dist))
            check_unary_dot(eval(f), rand(rng, x_dist, 100))
        end

        # Check that the dot notation works as expected for all of the binary functions in
        # Nabla for each permutation of scalar / array input.
        function check_binary_dot(f, x, y)
            x_, y_ = Leaf.(Tape(), (x, y))
            z_ = f.(x_, y_)
            z2_ = broadcast(f, x_, y_)
            @test z_.val == f.(x, y)
            @test ∇(z_, oneslike(z_.val))[x_] == ∇(z2_, oneslike(z2_.val))[x_]
            @test ∇(z_, oneslike(z_.val))[y_] == ∇(z2_, oneslike(z2_.val))[y_]
        end
        function check_binary_dot(f, x::∇Scalar, y::∇Scalar)
            x_, y_ = Leaf.(Tape(), (x, y))
            z_ = f.(x_, y_)
            @test ∇(z_)[x_] == ∇(broadcast(f, x_, y_))[x_]
            @test ∇(z_)[y_] == ∇(broadcast(f, x_, y_))[y_]
        end
        for (package, f) in Nabla.binary_sensitivities
            # TODO: More care needs to be taken to test the following.
            f in [:atan, :mod, :rem] && continue
            if hasdiffrule(package, f, 2)
                ∂f∂x, ∂f∂y = diffrule(package, f, :x, :y)
            else
                ∂f∂x, ∂f∂y = :∂f∂x, :∂f∂y
            end
            # TODO: Implement the edge cases for functions differentiable in only either
            # argument.
            (∂f∂x == :NaN || ∂f∂y == :NaN) && continue
            domain = domain2(eval(f))
            domain === nothing && error("Could not determine domain for $f.")
            (x_lb, x_ub), (y_lb, y_ub) = domain
            x_distr = Uniform(x_lb, x_ub)
            y_distr = Uniform(y_lb, y_ub)
            x = rand(rng, x_distr, 100)
            y = rand(rng, y_distr, 100)
            check_binary_dot(eval(f), x, y)
            check_binary_dot(eval(f), rand(rng, x_distr), y)
            check_binary_dot(eval(f), x, rand(rng, y_distr))
            check_binary_dot(eval(f), Ref(rand(rng, x_distr)), y)
            check_binary_dot(eval(f), x, Ref(rand(rng, y_distr)))
            check_binary_dot(eval(f), rand(rng, x_distr), rand(rng, y_distr))
        end
    end

    # Check that the number of allocations which happen in the reverse pass of `map` and
    # `broadcast` is invariant of the size of the data structure of which we are
    # mapping / broadcasting.
    for f in [:map, :broadcast]
        let
            @eval foo_small() = sum($f(tanh, Leaf(Tape(), randn(10, 10))))
            @eval foo_large() = sum($f(tanh, Leaf(Tape(), randn(10, 100))))
            @test allocs(@benchmark foo_small()) == allocs(@benchmark foo_large())
            @test allocs(@benchmark ∇(foo_small())) == allocs(@benchmark ∇(foo_large()))
        end
    end

    # #111
    let
        f(x) = sum(Float64[1,2,3] .* (x .+ Float64[3,2,1]))
        @test ∇(f)(Float64[1,2,3]) isa Tuple{Vector{Float64}}
        @test ∇(f; get_output=true)(Float64[1,2,3])[1].val == f(Float64[1,2,3])
    end

    # #117
    let
        f(x) = sum(x .+ [1,2,4] .* [4,2,1])
        @test ∇(f)(Float64[1,2,3])[1] == ones(Float64, 3)
        @test ∇(f; get_output=true)(Float64[1,2,3])[1].val == f(Float64[1,2,3])
    end

    # broadcasting literal_pow
    let
        f(x) = sum(x .^ 2)
        @test ∇(f)(Float64[1,2,3])[1] == Float64[2,4,6]
        @test ∇(f; get_output=true)(Float64[1,2,3])[1].val == f(Float64[1,2,3])
    end
end
