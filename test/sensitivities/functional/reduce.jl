@testset "Reduce" begin

    import Nabla.fmad

    # Check that `mapreduce`, `mapfoldl`and `mapfoldr` work as expected with all unary
    # functions, some composite functions which use FMAD under both `+` and `*`.
    let N = 3
        for functional in (mapreduce, mapfoldl, mapfoldr)

            # Sensitivities implemented in Base.
            for (f_, _, bounds, _) in Nabla.unary_sensitivities

                # Generate some data and get the function to be mapped.
                x = rand(Uniform(bounds[1], bounds[2]), N)
                f = eval(f_)

                # Test +.
                x_ = Leaf(Tape(), x)
                s = functional(f, +, x_)
                if Nabla.needs_output(f)
                    @test ∇(s)[x_] ≈ ∇.(f, Arg{1}, x, Base.map(f, x))
                else
                    @test ∇(s)[x_] ≈ ∇.(f, Arg{1}, x)
                end

                # # Test *.
                # x_ = Leaf(Tape(), x)
                # s = functional(f, +, x_)
                # @test s.val == functional(f, +, x)
                # @test Nabla.needs_output(f) ?
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
                @test ∇(s)[x_] ≈ map(x->fmad(f, (x,), Val{1}), x)

                # # Test *.
                # x_ = Leaf(Tape(), x)
                # s = functional(f, +, x_)
                # @test s.val == functional(f, +, x)
                # @test Nabla.needs_output(f) ?
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
            @test ∇(s)[x_] ≈ ones(x)

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
    let N = 5
        # Sensitivities implemented in Base.
        for (f_, _, bounds, _) in Nabla.unary_sensitivities

            # Generate some data and get the function to be mapped.
            x = rand(Uniform(bounds[1], bounds[2]), N)
            f = eval(Nabla, f_)

            # Test +.
            x_ = Leaf(Tape(), x)
            s = sum(f, x_)
            @test s.val == sum(f, x)
            @test Nabla.needs_output(f) ?
                ∇(s)[x_] ≈ ∇.(f, Arg{1}, x, map(f, x)) :
                ∇(s)[x_] ≈ ∇.(f, Arg{1}, x)
        end

        # Some composite functions.
        composite_functions = (x->5x, x->1 / (1 + x), x->10+x)
        for f in composite_functions

            # Generate some data.
            x = randn(N)

            # Test +.
            x_ = Leaf(Tape(), x)
            s = sum(f, x_)
            @test s.val == sum(f, x)
            @test ∇(s)[x_] ≈ map(x->fmad(f, (x,), Val{1}), x)
        end
    end
end
