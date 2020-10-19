@testset "Functional" begin
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
            return ∇(s, oneslike(unbox(s)))[x_] ≈ derivative_via_frule.(f, x)
        end
        @testset "$f" for f in UNARY_SCALAR_SENSITIVITIES
            domain = domain1(f)
            domain === nothing && error("Could not determine domain for $f.")
            x_dist = Uniform(domain...)
            x = rand(rng, x_dist, 100)
            @test check_unary_broadcast(f, x)
        end

        # Check that `broadcast` returns the correct gradient under each implemented binary
        # function.
        function check_binary_broadcast(f, x, y)
            tape = Tape()
            x_, y_ = Leaf(tape, x), Leaf(tape, y)
            s = broadcast(f, x_, y_)
            o = oneslike(unbox(s))
            ∇s = ∇(s, o)
#            ∇x = sum(broadcast((z, z̄, x, y)->∇(f, Arg{1}, nothing, z, z̄, x, y), unbox(s), o, x, y))
#            ∇y = broadcast((z, z̄, x, y)->∇(f, Arg{2}, nothing, z, z̄, x, y), unbox(s), o, x, y)
            @test broadcast(f, x, y) == unbox(s)
#            @test ∇s[x_] ≈ ∇x
#            @test ∇s[y_] ≈ ∇y
        end
        function check_binary_broadcast(f, x::Real, y)
            tape = Tape()
            x_, y_ = Leaf(tape, x), Leaf(tape, y)
            s = broadcast(f, x_, y_)
            o = oneslike(unbox(s))
            ∇s = ∇(s, o)
#            ∇x = sum(broadcast((z, z̄, x, y)->∇(f, Arg{1}, nothing, z, z̄, x, y), unbox(s), o, x, y))
#            ∇y = broadcast((z, z̄, x, y)->∇(f, Arg{2}, nothing, z, z̄, x, y), unbox(s), o, x, y)
            @test broadcast(f, x, y) == unbox(s)
#            @test ∇s[x_] ≈ ∇x
#            @test ∇s[y_] ≈ ∇y
        end
        function check_binary_broadcast(f, x, y::Real)
            tape = Tape()
            x_, y_ = Leaf(tape, x), Leaf(tape, y)
            s = broadcast(f, x_, y_)
            o = oneslike(unbox(s))
            ∇s = ∇(s, o)
#            ∇x = sum(broadcast((z, z̄, x, y)->∇(f, Arg{1}, nothing, z, z̄, x, y), unbox(s), o, x, y))
#            ∇y = broadcast((z, z̄, x, y)->∇(f, Arg{2}, nothing, z, z̄, x, y), unbox(s), o, x, y)
            @test broadcast(f, x, y) == unbox(s)
#            @test ∇s[x_] ≈ ∇x
#            @test ∇s[y_] ≈ ∇y
        end
        @testset "$f" for f in BINARY_SCALAR_SENSITIVITIES
            # TODO: Implement the edge cases for functions differentiable in only either
            # argument.
            f in  ONLY_DIFF_IN_SECOND_ARG_SENSITIVITIES && continue
            domain = domain2(f)
            domain === nothing && error("Could not determine domain for $f.")
            (x_lb, x_ub), (y_lb, y_ub) = domain
            x_dist, y_dist = Uniform(x_lb, x_ub), Uniform(y_lb, y_ub)
            x, y = rand(rng, x_dist, 100), rand(rng, y_dist, 100)
            check_binary_broadcast(f, x, y)
            check_binary_broadcast(f, rand(rng, x_dist), y)
            check_binary_broadcast(f, x, rand(rng, y_dist))
        end
        #
        let # Ternary functions (because it's useful to check I guess.)
            f = (x, y, z)->x * y + y * z + x * z
            x, y, z = randn(rng, 5), randn(rng, 5), randn(rng, 5)
            x_, y_, z_ = Leaf.(Tape(), (x, y, z))
            s_ = broadcast(f, x_, y_, z_)
            ∇s = ∇(s_, oneslike(unbox(s_)))
            @test unbox(s_) == broadcast(f, x, y, z)
            @test ∇s[x_] == getindex.(broadcast((x, y, z)->fmad(f, (x, y, z)), x, y, z), 1)
            @test ∇s[y_] == getindex.(broadcast((x, y, z)->fmad(f, (x, y, z)), x, y, z), 2)
            @test ∇s[z_] == getindex.(broadcast((x, y, z)->fmad(f, (x, y, z)), x, y, z), 3)
        end

        let
            x, y, tape = 5.0, randn(rng, 5), Tape()
            x_, y_ = Leaf(tape, x), Leaf(tape, y)
            z_ = x_ .+ y_
            z2_ = broadcast(+, x_, y_)
            @test unbox(z_) == x .+ y
            @test ∇(z_, oneslike(unbox(z_)))[x_] == ∇(z2_, oneslike(unbox(z2_)))[x_]
            @test ∇(z_, oneslike(unbox(z_)))[y_] == ∇(z2_, oneslike(unbox(z2_)))[y_]
        end
        let
            x, y, tape = randn(rng, 5), 5.0, Tape()
            x_, y_ = Leaf(tape, x), Leaf(tape, y)
            z_ = x_ * y_
            z2_ = broadcast(*, x_, y_)
            @test unbox(z_) == x .* y
            @test ∇(z_, oneslike(unbox(z_)))[x_] == ∇(z2_, oneslike(unbox(z2_)))[x_]
            @test ∇(z_, oneslike(unbox(z_)))[y_] == ∇(z2_, oneslike(unbox(z2_)))[y_]
        end
        let
            x, y, tape = randn(rng, 5), 5.0, Tape()
            x_, y_ = Leaf(tape, x), Leaf(tape, y)
            z_ = x_ .- y_
            z2_ = broadcast(-, x_, y_)
            @test unbox(z_) == x .- y
            @test ∇(z_, oneslike(unbox(z_)))[x_] == ∇(z2_, oneslike(unbox(z2_)))[x_]
            @test ∇(z_, oneslike(unbox(z_)))[y_] == ∇(z2_, oneslike(unbox(z2_)))[y_]
        end
        let
            x, y, tape = randn(rng, 5), 5.0, Tape()
            x_, y_ = Leaf(tape, x), Leaf(tape, y)
            z_ = x_ / y_
            z2_ = broadcast(/, x_, y_)
            @test unbox(z_) == x ./ y
            @test ∇(z_, oneslike(unbox(z_)))[x_] == ∇(z2_, oneslike(unbox(z2_)))[x_]
            @test ∇(z_, oneslike(unbox(z_)))[y_] ≈ ∇(z2_, oneslike(unbox(z2_)))[y_]
        end
        let
            x, y, tape = 5.0, randn(rng, 5), Tape()
            x_, y_ = Leaf(tape, x), Leaf(tape, y)
            z_ = x_ \ y_
            z2_ = broadcast(\, x_, y_)
            @test unbox(z_) == x .\ y
            @test ∇(z_, oneslike(unbox(z_)))[x_] ≈ ∇(z2_, oneslike(unbox(z2_)))[x_]
            @test ∇(z_, oneslike(unbox(z_)))[y_] == ∇(z2_, oneslike(unbox(z2_)))[y_]
        end

        # Check that dot notation works as expected for all unary function in Nabla for both
        # scalars and arrays.
        function check_unary_dot(f, x)
            x_ = Leaf(Tape(), x)
            z_ = f.(x_)
            z2_ = broadcast(f, x_)
            @test unbox(z_) == f.(x)
            @test ∇(z_, oneslike(unbox(z_)))[x_] == ∇(z2_, oneslike(unbox(z2_)))[x_]
        end
        function check_unary_dot(f, x::∇Scalar)
            x_ = Leaf(Tape(), x)
            z_ = f.(x_)
            @test unbox(z_) == f.(x)
            @test ∇(z_)[x_] == ∇(broadcast(f, x_))[x_]
        end
        for f in UNARY_SCALAR_SENSITIVITIES
            domain = domain1(f)
            domain === nothing && error("Could not determine domain for $f.")
            x_dist = Uniform(domain...)
            check_unary_dot(f, rand(rng, x_dist))
            check_unary_dot(f, rand(rng, x_dist, 100))
        end

        # Check that the dot notation works as expected for all of the binary functions in
        # Nabla for each permutation of scalar / array input.
        function check_binary_dot(f, x, y)
            x_, y_ = Leaf.(Tape(), (x, y))
            z_ = f.(x_, y_)
            z2_ = broadcast(f, x_, y_)
            @test unbox(z_) == f.(x, y)
            @test ∇(z_, oneslike(unbox(z_)))[x_] == ∇(z2_, oneslike(unbox(z2_)))[x_]
            @test ∇(z_, oneslike(unbox(z_)))[y_] == ∇(z2_, oneslike(unbox(z2_)))[y_]
        end
        function check_binary_dot(f, x::∇Scalar, y::∇Scalar)
            x_, y_ = Leaf.(Tape(), (x, y))
            z_ = f.(x_, y_)
            @test ∇(z_)[x_] == ∇(broadcast(f, x_, y_))[x_]
            @test ∇(z_)[y_] == ∇(broadcast(f, x_, y_))[y_]
        end
        for f in BINARY_SCALAR_SENSITIVITIES
            # TODO: More care needs to be taken to test the following.
            f in [atan, mod, rem] && continue
            # TODO: Implement the edge cases for functions differentiable in only either
            # argument.
            f in ONLY_DIFF_IN_SECOND_ARG_SENSITIVITIES && continue
            domain = domain2(f)
            domain === nothing && error("Could not determine domain for $f.")
            (x_lb, x_ub), (y_lb, y_ub) = domain
            x_distr = Uniform(x_lb, x_ub)
            y_distr = Uniform(y_lb, y_ub)
            x = rand(rng, x_distr, 100)
            y = rand(rng, y_distr, 100)
            check_binary_dot(f, x, y)
            check_binary_dot(f, rand(rng, x_distr), y)
            check_binary_dot(f, x, rand(rng, y_distr))
            check_binary_dot(f, Ref(rand(rng, x_distr)), y)
            check_binary_dot(f, x, Ref(rand(rng, y_distr)))
            check_binary_dot(f, rand(rng, x_distr), rand(rng, y_distr))
        end

        # test with other broadcast styles
        let
            a = Diagonal(ones(3))
            b = ones(3, 3)
            check_binary_dot(+, a, b)
            check_binary_dot(+, b, a)
            check_binary_dot(+, a, a)
        end
    end

    # Check that the number of allocations which happen in the reverse pass of `map` and
    # `broadcast` is invariant of the size of the data structure of which we are
    # mapping / broadcasting.
    @testset for f in [:map, :broadcast]
        let
            @eval foo_small() = sum($f(tanh, Leaf(Tape(), randn(10, 10))))
            @eval foo_large() = sum($f(tanh, Leaf(Tape(), randn(10, 100))))
            @test allocs(@benchmark foo_small()) ≈ allocs(@benchmark foo_large()) atol=1
            @test allocs(@benchmark ∇(foo_small())) ≈ allocs(@benchmark ∇(foo_large())) atol=1
        end
    end

    # #111
    let
        f(x) = sum(Float64[1,2,3] .* (x .+ Float64[3,2,1]))
        @test ∇(f)(Float64[1,2,3]) isa Tuple{Vector{Float64}}
        @test unbox(∇(f; get_output=true)(Float64[1,2,3])[1]) == f(Float64[1,2,3])
    end

    # #117
    let
        f(x) = sum(x .+ [1,2,4] .* [4,2,1])
        @test ∇(f)(Float64[1,2,3])[1] == ones(Float64, 3)
        @test unbox(∇(f; get_output=true)(Float64[1,2,3])[1]) == f(Float64[1,2,3])
    end

    # broadcasting literal_pow
    let
        f(x) = sum(x .^ 2)
        @test ∇(f)(Float64[1,2,3])[1] == Float64[2,4,6]
        @test unbox(∇(f; get_output=true)(Float64[1,2,3])[1]) == f(Float64[1,2,3])
    end

    # fused broadcasting with different styles
    let
        f(x) = sum(Symmetric(x) .+ 0.0001 .* Diagonal(ones(size(x, 1))))
        a = rand(3, 3)
        a += transpose(a)
        @test ∇(f)(a)[1] == Float64[1 2 2; 0 1 2; 0 0 1]
        @test unbox(∇(f; get_output=true)(a)[1]) == f(a)
    end
end

struct CoolArray{T} <: AbstractVector{T}
    x::Vector{T}
end
Base.map(f, x::CoolArray, y::AbstractArray) = map(f, x.x, y)
@testset "Issue #136" begin
    # This doesn't even involve Nabla at all; we're just making sure that we aren't
    # introducing method ambiguities with our `map` overloading that can interfere with
    # other packages that extend `map`, e.g. StaticArrays
    @test map(-, CoolArray([1,2,3]), [1,2,3]) == [0,0,0]

    # `map` with lots of inputs
    for i = 1:10
        args = Any[1:3 for _ = 1:20]
        args[i] = Leaf(Tape(), 1:3)
        x = map(+, args...)
        @test x isa Branch{Vector{Int}}
        @test unbox(x) == [20,40,60]
    end
end
