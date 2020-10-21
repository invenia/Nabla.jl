@testset "Reduce" begin
    let rng = MersenneTwister(123456)
        # Check that `mapreduce`, `mapfoldl`and `mapfoldr` work as expected with all unary
        # functions, some composite functions which use FMAD under both `+` and `*`.
        let N = 3
            for functional in (mapreduce, mapfoldl, mapfoldr)

                # Sensitivities implemented in Base.
                for f in UNARY_SCALAR_SENSITIVITIES

                    # Generate some data and get the function to be mapped.
                    domain = domain1(f)
                    domain === nothing && error("Could not determine domain for $f.")
                    lb, ub = domain
                    x = rand(rng, N) .* (ub - lb) .+ lb

                    # Test +.
                    x_ = Leaf(Tape(), x)
                    s = functional(f, +, x_)
                    @test ∇(s)[x_] ≈ derivative_via_frule.(f, x)
                end

                # Some composite sensitivities.
               composite_functions = (x->5x, x->1 / (1 + x), x->10+x)
                for f in composite_functions

                    # Generate some data.
                    x = randn(rng, 100)

                    # Test +.
                    x_ = Leaf(Tape(), x)
                    s = functional(f, +, x_)
                    @test unbox(s) ≈ functional(f, +, x)
                    @test ∇(s)[x_] ≈ map(xn->ForwardDiff.derivative(f, xn), x)
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
                x = randn(rng, 100)
                x_ = Leaf(Tape(), x)
                s = functional(+, x_)
                @test unbox(s) == functional(+, x)
                @test ∇(s)[x_] ≈ oneslike(100)
            end
        end

        # Check that `sum` and `prod` work as expected under all implemented unary functions
        # and some composite functions which use FMAD.
        let N = 5
            # Sensitivities implemented in Base.
            for f in UNARY_SCALAR_SENSITIVITIES
                # Generate some data and get the function to be mapped.
                domain = domain1(f)
                domain === nothing && error("Could not determine domain for $f.")
                lb, ub = domain
                x = rand(rng, N) .* (ub .- lb) .+ lb

                # Test +.
                x_ = Leaf(Tape(), x)
                s = sum(f, x_)
                @test unbox(s) == sum(f, x)
                @test ∇(s)[x_] ≈ derivative_via_frule.(f, x)
            end

            # Some composite functions.
            composite_functions = (x->5x, x->1 / (1 + x), x->10+x)
            for f in composite_functions

                # Generate some data.
                x = randn(rng, N)

                # Test +.
                x_ = Leaf(Tape(), x)
                s = sum(f, x_)
                @test unbox(s) == sum(f, x)
                @test ∇(s)[x_] ≈ map(xn->ForwardDiff.derivative(f, xn), x)
            end
        end
    end
end
