@testset "sensitivities/functional" begin

    import Nabla.DiffBase.fmad

    # Simple test with known gradient results.
    x = Leaf(Tape(), [1, 2, 3, 4, 5])
    s = 5 * mapreduce(abs2, +, x)
    @test ∇(s)[x] == 5 * [2, 4, 6, 8, 10]

    function check_unary_sum(f, x)
        x_ = Leaf(Tape(), x)
        s = mapreduce(f, +, x_)
        return DiffBase.needs_output(f) ?
            ∇(s)[x_] == ∇.(f, Arg{1}, x, Base.map(f, x)) :
            ∇(s)[x_] == ∇.(f, Arg{1}, x)
    end

    # Iterate over all unary functions and check they work correctly with mapreduce.
    for (f, _, bounds, _) in DiffBase.unary_sensitivities
        x = rand(Uniform(bounds[1], bounds[2]), 100)
        @test check_unary_sum(eval(current_module(), f), x)
    end

    # Check that the FMAD implementation is doing it's job on functions for which we do not
    # have gradient implementations.
    let # Unary functions.
        fs = (x->5x, x->1 / (1 + x), x->10+x)
        for f in fs
            x = randn(5)
            x_ = Leaf(Tape(), x)
            s_ = mapreduce(f, +, x_)
            @test s_.val == mapreduce(f, +, x)
            @test ∇(s_)[x_] == map(x->DiffBase.fmad(f, (x,), Val{1}), x)
        end
    end

    # mapreducedim on a single-dimensional array should be consistent with mapreduce.
    x = Leaf(Tape(), [1, 2, 3, 4, 5])
    s = 5 * mapreducedim(abs2, +, x, 1)[1]
    @test ∇(s)[x] == 5 * [2, 4, 6, 8, 10]

    # mapreducedim on a two-dimensional array when reduced over a single dimension should
    # give different results to mapreduce over the same array.
    x2_ = reshape([1, 2, 3, 4,], (2, 2))
    x2 = Leaf(Tape(), x2_)
    s = mapreducedim(abs2, +, x2, 1)
    @test ∇(s)[x] == 2 * x2_

    # Simple test for mapping under the identity.
    x = Leaf(Tape(), [1, 2, 3, 4, 5])
    s = map(identity, x)
    @test s.val == [1, 2, 3, 4, 5]
    @test ∇(s)[x] == [1, 1, 1, 1, 1]

    # Check that `map` returns the correct gradient under a unary function f.
    function check_unary_map(f, x)
        x_ = Leaf(Tape(), x)
        s = map(f, x_)
        return DiffBase.needs_output(f) ?
            ∇(s)[x_] == ∇.(f, Arg{1}, x, Base.map(f, x)) :
            ∇(s)[x_] == ∇.(f, Arg{1}, x)
    end
    for (f, _, bounds, _) in DiffBase.unary_sensitivities
        x = rand(Uniform(bounds[1], bounds[2]), 100)
        @test check_unary_map(eval(current_module(), f), x)
    end

    # Check that `map` returns the correct gradient under each implemented binary function.
    function check_binary_map(f, x, y)
        tape = Tape()
        x_, y_ = Leaf(tape, x), Leaf(tape, y)
        s = map(f, x_, y_)
        ∇s = ∇(s)
        ∇x = map((z, z̄, x, y)->∇(f, Arg{1}, nothing, z, z̄, x, y), s.val, ones(s.val), x, y)
        ∇y = map((z, z̄, x, y)->∇(f, Arg{2}, nothing, z, z̄, x, y), s.val, ones(s.val), x, y)
        return map(f, x, y) == s.val && ∇s[x_] == ∇x && ∇s[y_] == ∇y
    end
    for (f, _, _, x_bounds, y_bounds) in DiffBase.binary_sensitivities
        x = rand(Uniform(x_bounds[1], x_bounds[2]), 100)
        y = rand(Uniform(y_bounds[1], y_bounds[2]), 100)
        @test check_binary_map(eval(current_module(), f), x, y)
    end

    # Check that map returns the correct gradients for unary, binary and tenary functions
    # that do not have explicit implementations.
    let # Unary functions.
        fs = (x->5x, x->1 / (1 + x), x->10+x)
        for f in fs
            x = randn(5)
            x_ = Leaf(Tape(), x)
            s_ = map(f, x_)
            @test s_.val == map(f, x)
            @test ∇(s_)[x_] == getindex.(map((x)->fmad(f, (x,)), x), 1)
        end
    end
    let # Binary functions.
        fs = ((x, y)->x + y, (x, y)->x + x * y, (x, y)->tanh(x) * sin(y))
        for f in fs
            x, y = randn(5), randn(5)
            x_, y_ = Leaf.(Tape(), (x, y))
            s_ = map(f, x_, y_)
            @test s_.val == map(f, x, y)
            @test ∇(s_)[x_] == getindex.(map((x, y)->fmad(f, (x, y)), x, y), 1)
            @test ∇(s_)[y_] == getindex.(map((x, y)->fmad(f, (x, y)), x, y), 2)
        end
    end
    let # Ternary functions (because it's useful to check I guess.)
        f = (x, y, z)->x * y + y * z + x * z
        x, y, z = randn(5), randn(5), randn(5)
        x_, y_, z_ = Leaf.(Tape(), (x, y, z))
        s_ = map(f, x_, y_, z_)
        @test s_.val == map(f, x, y, z)
        @test ∇(s_)[x_] == getindex.(map((x, y, z)->fmad(f, (x, y, z)), x, y, z), 1)
        @test ∇(s_)[y_] == getindex.(map((x, y, z)->fmad(f, (x, y, z)), x, y, z), 2)
        @test ∇(s_)[z_] == getindex.(map((x, y, z)->fmad(f, (x, y, z)), x, y, z), 3)
    end

    # Check that `broadcast` returns the correct gradient under the defined unary functions.
    function check_unary_broadcast(f, x)
        x_ = Leaf(Tape(), x)
        s = broadcast(f, x_)
        return DiffBase.needs_output(f) ?
            ∇(s)[x_] == ∇.(f, Arg{1}, x, Base.map(f, x)) :
            ∇(s)[x_] == ∇.(f, Arg{1}, x)
    end
    for (f, _, bounds, _) in DiffBase.unary_sensitivities
        x = rand(Uniform(bounds[1], bounds[2]), 100)
        @test check_unary_broadcast(eval(current_module(), f), x)
    end

    # Check that `broadcast` returns the correct gradient under each implemented binary function.
    function check_binary_broadcast(f, x, y)
        tape = Tape()
        x_, y_ = Leaf(tape, x), Leaf(tape, y)
        s = broadcast(f, x_, y_)
        ∇s = ∇(s)
        ∇x = broadcast((z, z̄, x, y)->∇(f, Arg{1}, nothing, z, z̄, x, y), s.val, ones(s.val), x, y)
        ∇y = broadcast((z, z̄, x, y)->∇(f, Arg{2}, nothing, z, z̄, x, y), s.val, ones(s.val), x, y)
        @test broadcast(f, x, y) == s.val
        @test ∇s[x_] == ∇x
        @test ∇s[y_] == ∇y
    end
    function check_binary_broadcast(f, x::Real, y)
        tape = Tape()
        x_, y_ = Leaf(tape, x), Leaf(tape, y)
        s = broadcast(f, x_, y_)
        ∇s = ∇(s)
        ∇x = sum(broadcast((z, z̄, x, y)->∇(f, Arg{1}, nothing, z, z̄, x, y), s.val, ones(s.val), x, y))
        ∇y = broadcast((z, z̄, x, y)->∇(f, Arg{2}, nothing, z, z̄, x, y), s.val, ones(s.val), x, y)
        @test broadcast(f, x, y) == s.val
        @test ∇s[x_] == ∇x
        @test ∇s[y_] == ∇y
    end
    function check_binary_broadcast(f, x, y::Real)
        tape = Tape()
        x_, y_ = Leaf(tape, x), Leaf(tape, y)
        s = broadcast(f, x_, y_)
        ∇s = ∇(s)
        ∇x = broadcast((z, z̄, x, y)->∇(f, Arg{1}, nothing, z, z̄, x, y), s.val, ones(s.val), x, y)
        ∇y = sum(broadcast((z, z̄, x, y)->∇(f, Arg{2}, nothing, z, z̄, x, y), s.val, ones(s.val), x, y))
        @test broadcast(f, x, y) == s.val
        @test ∇s[x_] == ∇x
        @test ∇s[y_] == ∇y
    end
    for (f, _, _, x_bounds, y_bounds) in DiffBase.binary_sensitivities
        x_distr = Uniform(x_bounds[1], x_bounds[2])
        y_distr = Uniform(y_bounds[1], y_bounds[2])
        x = rand(x_distr, 100)
        y = rand(y_distr, 100)
        check_binary_broadcast(eval(current_module(), f), x, y)
        check_binary_broadcast(eval(current_module(), f), rand(x_distr), y)
        check_binary_broadcast(eval(current_module(), f), x, rand(y_distr))
    end

    let # Ternary functions (because it's useful to check I guess.)
        f = (x, y, z)->x * y + y * z + x * z
        x, y, z = randn(5), randn(5), randn(5)
        x_, y_, z_ = Leaf.(Tape(), (x, y, z))
        s_ = broadcast(f, x_, y_, z_)
        @test s_.val == broadcast(f, x, y, z)
        @test ∇(s_)[x_] == getindex.(broadcast((x, y, z)->fmad(f, (x, y, z)), x, y, z), 1)
        @test ∇(s_)[y_] == getindex.(broadcast((x, y, z)->fmad(f, (x, y, z)), x, y, z), 2)
        @test ∇(s_)[z_] == getindex.(broadcast((x, y, z)->fmad(f, (x, y, z)), x, y, z), 3)
    end

    let
        x, y, tape = 5.0, randn(5), Tape()
        x_, y_ = Leaf(tape, x), Leaf(tape, y)
        z_ = x_ + y_
        @test z_.val == x + y
        @test ∇(z_)[x_] == ∇(broadcast(+, x_, y_))[x_]
        @test ∇(z_)[y_] == ∇(broadcast(+, x_, y_))[y_]
    end
    let
        x, y, tape = randn(5), 5.0, Tape()
        x_, y_ = Leaf(tape, x), Leaf(tape, y)
        z_ = x_ * y_
        @test z_.val == x * y
        @test ∇(z_)[x_] == ∇(broadcast(*, x_, y_))[x_]
        @test ∇(z_)[y_] == ∇(broadcast(*, x_, y_))[y_]
    end
    let
        x, y, tape = randn(5), 5.0, Tape()
        x_, y_ = Leaf(tape, x), Leaf(tape, y)
        z_ = x_ - y_
        @test z_.val == x - y
        @test ∇(z_)[x_] == ∇(broadcast(-, x_, y_))[x_]
        @test ∇(z_)[y_] == ∇(broadcast(-, x_, y_))[y_]
    end
    let
        x, y, tape = randn(5), 5.0, Tape()
        x_, y_ = Leaf(tape, x), Leaf(tape, y)
        z_ = x_ / y_
        @test z_.val == x / y
        @test ∇(z_)[x_] == ∇(broadcast(/, x_, y_))[x_]
        @test ∇(z_)[y_] == ∇(broadcast(/, x_, y_))[y_]
    end
    let
        x, y, tape = 5.0, randn(5), Tape()
        x_, y_ = Leaf(tape, x), Leaf(tape, y)
        z_ = x_ \ y_
        @test z_.val == x \ y
        @test ∇(z_)[x_] == ∇(broadcast(\, x_, y_))[x_]
        @test ∇(z_)[y_] == ∇(broadcast(\, x_, y_))[y_]
    end

    # Check that dot notation works as expected for all unary function in DiffBase for both
    # scalars and arrays.
    function check_unary_dot(f, x)
        x_ = Leaf(Tape(), x)
        z_ = f.(x_)
        @test z_.val == f.(x)
        @test ∇(z_)[x_] == ∇(broadcast(f, x_))[x_]
    end
    for (f, _, bounds, _) in DiffBase.unary_sensitivities
        distr = Uniform(bounds[1], bounds[2])
        check_unary_dot(eval(current_module(), f), rand(distr))
        check_unary_dot(eval(current_module(), f), rand(distr, 100))
    end

    # Check that the dot notation works as expected for all of the binary functions in
    # DiffBase for each permutation of scalar / array input.
    function check_binary_dot(f, x, y)
        x_, y_ = Leaf.(Tape(), (x, y))
        z_ = f.(x_, y_)
        @test z_.val == f.(x, y)
        @test ∇(z_)[x_] == ∇(broadcast(f, x_, y_))[x_]
        @test ∇(z_)[y_] == ∇(broadcast(f, x_, y_))[y_]
    end
    for (f, _, _, x_bounds, y_bounds) in DiffBase.binary_sensitivities
        x_distr = Uniform(x_bounds[1], x_bounds[2])
        y_distr = Uniform(y_bounds[1], y_bounds[2])
        x = rand(x_distr, 100)
        y = rand(y_distr, 100)
        check_binary_dot(eval(current_module(), f), x, y)
        check_binary_dot(eval(current_module(), f), rand(x_distr), y)
        check_binary_dot(eval(current_module(), f), x, rand(y_distr))
        check_binary_dot(eval(current_module(), f), rand(x_distr), rand(y_distr))
    end
end
