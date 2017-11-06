using SpecialFunctions, DiffRules

@testset "Functional" begin

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
        return Nabla.needs_output(f) ?
            ∇(s, ones(s.val))[x_] ≈ ∇.(f, Arg{1}, x, Base.map(f, x)) :
            ∇(s, ones(s.val))[x_] ≈ ∇.(f, Arg{1}, x)
    end
    for (package, f) in Nabla.unary_sensitivities
        domain = domain1(eval(f))
        isnull(domain) && error("Could not determine domain for $f.")
        x = rand(Uniform(get(domain)...), 100)
        @test check_unary_broadcast(eval(f), x)
    end

    # Check that `broadcast` returns the correct gradient under each implemented binary
    # function.
    function check_binary_broadcast(f, x, y)
        tape = Tape()
        x_, y_ = Leaf(tape, x), Leaf(tape, y)
        s = broadcast(f, x_, y_)
        ∇s = ∇(s, ones(s.val))
        ∇x = broadcast((z, z̄, x, y)->∇(f, Arg{1}, nothing, z, z̄, x, y),
                       s.val, ones(s.val), x, y)
        ∇y = broadcast((z, z̄, x, y)->∇(f, Arg{2}, nothing, z, z̄, x, y),
                       s.val, ones(s.val), x, y)
        @test broadcast(f, x, y) == s.val
        @test ∇s[x_] ≈ ∇x
        @test ∇s[y_] ≈ ∇y
    end
    function check_binary_broadcast(f, x::Real, y)
        tape = Tape()
        x_, y_ = Leaf(tape, x), Leaf(tape, y)
        s = broadcast(f, x_, y_)
        ∇s = ∇(s, ones(s.val))
        ∇x = sum(broadcast((z, z̄, x, y)->∇(f, Arg{1}, nothing, z, z̄, x, y),
                           s.val, ones(s.val), x, y))
        ∇y = broadcast((z, z̄, x, y)->∇(f, Arg{2}, nothing, z, z̄, x, y),
                       s.val, ones(s.val), x, y)
        @test broadcast(f, x, y) == s.val
        @test ∇s[x_] ≈ ∇x
        @test ∇s[y_] ≈ ∇y
    end
    function check_binary_broadcast(f, x, y::Real)
        tape = Tape()
        x_, y_ = Leaf(tape, x), Leaf(tape, y)
        s = broadcast(f, x_, y_)
        ∇s = ∇(s, ones(s.val))
        ∇x = broadcast((z, z̄, x, y)->∇(f, Arg{1}, nothing, z, z̄, x, y),
                       s.val, ones(s.val), x, y)
        ∇y = sum(broadcast((z, z̄, x, y)->∇(f, Arg{2}, nothing, z, z̄, x, y),
                           s.val, ones(s.val), x, y))
        @test broadcast(f, x, y) == s.val
        @test ∇s[x_] ≈ ∇x
        @test ∇s[y_] ≈ ∇y
    end
    for (package, f) in Nabla.binary_sensitivities
        # TODO: More care needs to be taken to test the following.
        f in [:atan2, :mod, :rem] && continue
        ∂f∂x, ∂f∂y = DiffRules.diffrule(package, f, :x, :y)
        # TODO: Implement the edge cases for functions differentiable in only either
        # argument.
        (∂f∂x == :NaN || ∂f∂y == :NaN) && continue
        domain = domain2(eval(f))
        isnull(domain) && error("Could not determine domain for $f.")
        (x_lb, x_ub), (y_lb, y_ub) = get(domain)
        x_distr, y_distr = Uniform(x_lb, x_ub), Uniform(y_lb, y_ub)
        x = rand(x_distr, 100)
        y = rand(y_distr, 100)
        check_binary_broadcast(eval(f), x, y)
        check_binary_broadcast(eval(f), rand(x_distr), y)
        check_binary_broadcast(eval(f), x, rand(y_distr))
    end
    #
    let # Ternary functions (because it's useful to check I guess.)
        f = (x, y, z)->x * y + y * z + x * z
        x, y, z = randn(5), randn(5), randn(5)
        x_, y_, z_ = Leaf.(Tape(), (x, y, z))
        s_ = broadcast(f, x_, y_, z_)
        ∇s = ∇(s_, ones(s_.val))
        @test s_.val == broadcast(f, x, y, z)
        @test ∇s[x_] == getindex.(broadcast((x, y, z)->fmad(f, (x, y, z)), x, y, z), 1)
        @test ∇s[y_] == getindex.(broadcast((x, y, z)->fmad(f, (x, y, z)), x, y, z), 2)
        @test ∇s[z_] == getindex.(broadcast((x, y, z)->fmad(f, (x, y, z)), x, y, z), 3)
    end

    let
        x, y, tape = 5.0, randn(5), Tape()
        x_, y_ = Leaf(tape, x), Leaf(tape, y)
        z_ = x_ + y_
        z2_ = broadcast(+, x_, y_)
        @test z_.val == x + y
        @test ∇(z_, ones(z_.val))[x_] == ∇(z2_, ones(z2_.val))[x_]
        @test ∇(z_, ones(z_.val))[y_] == ∇(z2_, ones(z2_.val))[y_]
    end
    let
        x, y, tape = randn(5), 5.0, Tape()
        x_, y_ = Leaf(tape, x), Leaf(tape, y)
        z_ = x_ * y_
        z2_ = broadcast(*, x_, y_)
        @test z_.val == x * y
        @test ∇(z_, ones(z_.val))[x_] == ∇(z2_, ones(z2_.val))[x_]
        @test ∇(z_, ones(z_.val))[y_] == ∇(z2_, ones(z2_.val))[y_]
    end
    let
        x, y, tape = randn(5), 5.0, Tape()
        x_, y_ = Leaf(tape, x), Leaf(tape, y)
        z_ = x_ - y_
        z2_ = broadcast(-, x_, y_)
        @test z_.val == x - y
        @test ∇(z_, ones(z_.val))[x_] == ∇(z2_, ones(z2_.val))[x_]
        @test ∇(z_, ones(z_.val))[y_] == ∇(z2_, ones(z2_.val))[y_]
    end
    let
        x, y, tape = randn(5), 5.0, Tape()
        x_, y_ = Leaf(tape, x), Leaf(tape, y)
        z_ = x_ / y_
        z2_ = broadcast(/, x_, y_)
        @test z_.val == x / y
        @test ∇(z_, ones(z_.val))[x_] == ∇(z2_, ones(z2_.val))[x_]
        @test ∇(z_, ones(z_.val))[y_] == ∇(z2_, ones(z2_.val))[y_]
    end
    let
        x, y, tape = 5.0, randn(5), Tape()
        x_, y_ = Leaf(tape, x), Leaf(tape, y)
        z_ = x_ \ y_
        z2_ = broadcast(\, x_, y_)
        @test z_.val == x \ y
        @test ∇(z_, ones(z_.val))[x_] == ∇(z2_, ones(z2_.val))[x_]
        @test ∇(z_, ones(z_.val))[y_] == ∇(z2_, ones(z2_.val))[y_]
    end

    # Check that dot notation works as expected for all unary function in Nabla for both
    # scalars and arrays.
    function check_unary_dot(f, x)
        x_ = Leaf(Tape(), x)
        z_ = f.(x_)
        z2_ = broadcast(f, x_)
        @test z_.val == f.(x)
        @test ∇(z_, ones(z_.val))[x_] == ∇(z2_, ones(z2_.val))[x_]
    end
    function check_unary_dot(f, x::∇Scalar)
        x_ = Leaf(Tape(), x)
        z_ = f.(x_)
        @test z_.val == f.(x)
        @test ∇(z_)[x_] == ∇(broadcast(f, x_))[x_]
    end
    for (package, f) in Nabla.unary_sensitivities
        domain = domain1(eval(f))
        isnull(domain) && error("Could not determine domain for $f.")
        distr = Uniform(get(domain)...)
        check_unary_dot(eval(f), rand(distr))
        check_unary_dot(eval(f), rand(distr, 100))
    end

    # Check that the dot notation works as expected for all of the binary functions in
    # Nabla for each permutation of scalar / array input.
    function check_binary_dot(f, x, y)
        x_, y_ = Leaf.(Tape(), (x, y))
        z_ = f.(x_, y_)
        z2_ = broadcast(f, x_, y_)
        @test z_.val == f.(x, y)
        @test ∇(z_, ones(z_.val))[x_] == ∇(z2_, ones(z2_.val))[x_]
        @test ∇(z_, ones(z_.val))[y_] == ∇(z2_, ones(z2_.val))[y_]
    end
    function check_binary_dot(f, x::∇Scalar, y::∇Scalar)
        x_, y_ = Leaf.(Tape(), (x, y))
        z_ = f.(x_, y_)
        @test ∇(z_)[x_] == ∇(broadcast(f, x_, y_))[x_]
        @test ∇(z_)[y_] == ∇(broadcast(f, x_, y_))[y_]
    end
    for (package, f) in Nabla.binary_sensitivities
        # TODO: More care needs to be taken to test the following.
        f in [:atan2, :mod, :rem] && continue
        ∂f∂x, ∂f∂y = DiffRules.diffrule(package, f, :x, :y)
        # TODO: Implement the edge cases for functions differentiable in only either
        # argument.
        (∂f∂x == :NaN || ∂f∂y == :NaN) && continue
        domain = domain2(eval(f))
        isnull(domain) && error("Could not determine domain for $f.")
        (x_lb, x_ub), (y_lb, y_ub) = get(domain)
        x_distr, y_distr = Uniform(x_lb, x_ub), Uniform(y_lb, y_ub)
        x = rand(x_distr, 100)
        y = rand(y_distr, 100)
        check_binary_dot(eval(f), x, y)
        check_binary_dot(eval(f), rand(x_distr), y)
        check_binary_dot(eval(f), x, rand(y_distr))
        check_binary_dot(eval(f), rand(x_distr), rand(y_distr))
    end
end
