@testset "sensitivities/functional/reduce" begin

    import Nabla.DiffBase.fmad

    # Check that `mapreduce`, `mapfoldl`and `mapfoldr` work as expected with all unary
    # functions, some composite functions which use FMAD under both `+` and `*`.
    let
        for functional in (mapreduce, mapfoldl, mapfoldr)

            # Sensitivities implemented in Base.
            for (f_, _, bounds, _) in DiffBase.unary_sensitivities

                # Generate some data and get the function to be mapped.
                x = rand(Uniform(bounds[1], bounds[2]), 100)
                f = eval(DiffBase, f_)

                # Test +.
                x_ = Leaf(Tape(), x)
                s = functional(f, +, x_)
                @test s.val == functional(f, +, x)
                @test DiffBase.needs_output(f) ?
                    ∇(s)[x_] == ∇.(f, Arg{1}, x, Base.map(f, x)) :
                    ∇(s)[x_] == ∇.(f, Arg{1}, x)

                # # Test *.
                # x_ = Leaf(Tape(), x)
                # s = functional(f, +, x_)
                # @test s.val == functional(f, +, x)
                # @test DiffBase.needs_output(f) ?
                #     ∇(s)[x_] == ∇.(f, Arg{1}, x, Base.map(f, x)) :
                #     ∇(s)[x_] == ∇.(f, Arg{1}, x)
            end

            # Some composite sensitivities.
            composite_functions = (x->5x, x->1 / (1 + x), x->10+x)
            for f in composite_functions

                # Generate some data.
                x = randn(100)

                # Test +.
                x_ = Leaf(Tape(), x)
                s = functional(f, +, x_)
                @test s.val == functional(f, +, x)
                @test ∇(s)[x_] == map(x->DiffBase.fmad(f, (x,), Val{1}), x)

                # # Test *.
                # x_ = Leaf(Tape(), x)
                # s = functional(f, +, x_)
                # @test s.val == functional(f, +, x)
                # @test DiffBase.needs_output(f) ?
                #     ∇(s)[x_] == ∇.(f, Arg{1}, x, Base.map(f, x)) :
                #     ∇(s)[x_] == ∇.(f, Arg{1}, x)
            end
        end
    end

    # Check that something sensible (if rather slow) happens for some arbitrary `op`s.
    let

    end

    # Check that `reduce`, `foldl` and `foldr` work as expected for `+` and `*`.
    let
        for functional in (reduce, foldl, foldr)

            # Test `+`.
            x = randn(100)
            x_ = Leaf(Tape(), x)
            s = functional(+, x_)
            @test s.val == functional(+, x)
            @test ∇(s)[x_] == ones(x)

            # # Test `*`.
            # x = randn(100)
            # x_ = Leaf(Tape(), x)
            # s = functional(*, x_)
            # @test s.val == functional(*, x)
            # @test ...
        end
    end

    # Check that `sum` and `prod` work as expected under all implemented unary functions
    # and some composite functions which use FMAD.
    let
        # Sensitivities implemented in Base.
        for (f_, _, bounds, _) in DiffBase.unary_sensitivities

            # Generate some data and get the function to be mapped.
            x = rand(Uniform(bounds[1], bounds[2]), 100)
            f = eval(DiffBase, f_)

            # Test +.
            x_ = Leaf(Tape(), x)
            s = sum(f, x_)
            @test s.val == sum(f, x)
            @test DiffBase.needs_output(f) ?
                ∇(s)[x_] == ∇.(f, Arg{1}, x, Base.map(f, x)) :
                ∇(s)[x_] == ∇.(f, Arg{1}, x)
        end

        # Some composite functions.
        composite_functions = (x->5x, x->1 / (1 + x), x->10+x)
        for f in composite_functions

            # Generate some data.
            x = randn(100)

            # Test +.
            x_ = Leaf(Tape(), x)
            s = sum(f, x_)
            @test s.val == sum(f, x)
            @test ∇(s)[x_] == map(x->DiffBase.fmad(f, (x,), Val{1}), x)
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

end
