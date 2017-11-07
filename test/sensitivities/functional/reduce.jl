@testset "Reduce" begin

    import Nabla.fmad

    # Check that `mapreduce`, `mapfoldl`and `mapfoldr` work as expected with all unary
    # functions, some composite functions which use FMAD under both `+` and `*`.
    let N = 3
        for functional in (mapreduce, mapfoldl, mapfoldr)

            # Sensitivities implemented in Base.
            for (package, f) in Nabla.unary_sensitivities

                # Generate some data and get the function to be mapped.
                f = eval(f)
                domain = domain1(f)
                isnull(domain) && error("Could not determine domain for $f.")
                x = rand(Uniform(get(domain)...), N)

                # Test +.
                x_ = Leaf(Tape(), x)
                s = functional(f, +, x_)
                @test ∇(s)[x_] ≈ ∇.(f, Arg{1}, x)
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
        for (package, f) in Nabla.unary_sensitivities

            # Generate some data and get the function to be mapped.
            f = eval(f)
            domain = domain1(f)
            isnull(domain) && error("Could not determine domain for $f.")
            x = rand(Uniform(get(domain)...), N)

            # Test +.
            x_ = Leaf(Tape(), x)
            s = sum(f, x_)
            @test s.val == sum(f, x)
            @test ∇(s)[x_] ≈ ∇.(f, Arg{1}, x)
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
