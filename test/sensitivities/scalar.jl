@testset "Scalar domains" begin
    @test in_domain(sin, 10.)
    @test in_domain(cos, 10.)
    @test !in_domain(acos, 10.)
    @test !in_domain(asin, 10.)
    @test domain1(sin) == (minimum(points), maximum(points))
    @test domain1(log) == (minimum(points[points .> 0]), maximum(points))
    @test domain1(acos) == (minimum(points[points .> -1]),
                             maximum(points[points .< 1]))
    @test domain2((+)) == ((minimum(points), maximum(points)),
                            (minimum(points), maximum(points)))
    @test domain2((^)) == ((minimum(points[points .> 0]), maximum(points)),
                            (minimum(points), maximum(points)))
    @test domain2(beta) == ((minimum(points[points .> 0]), maximum(points)),
                                 (minimum(points[points .> 0]), maximum(points)))
end

@testset "Scalar" begin
    let v = 1.0, ȳ = 5.0, z̄ = 4.0, rng = MersenneTwister(123456)
        let
            @test ∇(identity, Arg{1}, 5.0, 4.0, 3.0, 2.0) == 3.0
            @test ∇(identity, Arg{1}, 5) == 1
            @test ∇(identity, Arg{1}, 5.0) == 1.0
        end

        unary_check(f, x) = check_errs(f, ȳ, x, v)
        @testset "$f" for f in UNARY_SCALAR_SENSITIVITIES
            domain = domain1(f)
            domain === nothing && error("Could not determine domain for $f.")
            lb, ub = domain
            randx = () -> rand(rng) * (ub - lb) + lb

            for _ in 1:10
                @test unary_check(f, randx())
            end
        end

        @testset "$f" for f in BINARY_SCALAR_SENSITIVITIES
            if f in ONLY_DIFF_IN_SECOND_ARG_SENSITIVITIES
                # First argument is not differentiable, it is integer-valued.
                domain = domain1(y -> f(0, y))
                domain === nothing && error("Could not determine domain for $f.")
                lb, ub = domain
                randx = () -> rand(rng, 0:5)
                randy = () -> rand(rng) * (ub - lb) + lb

                for _ in 1:10
                    x = randx()
                    @test check_errs(y -> f(x, y), ȳ, randy(), v)
                end
            else  # Both arguments are differentiable
                domain = domain2(f)
                domain === nothing && error("Could not determine domain for $f.")
                (x_lb, x_ub), (y_lb, y_ub) = domain
                randx = () -> rand(rng) * (x_ub - x_lb) + x_lb
                randy = () -> rand(rng) * (y_ub - y_lb) + y_lb

                for _ in 1:10
                    @test check_errs(f, z̄, (randx(), randy()), (v, v))
                end
            end
        end

        # Test whether the exponentiation amibiguity is resolved.
        @test ∇(x -> x^2)(1) == (2.0,)
    end

    @testset "float" begin
        # Scalars
        x_ = 4
        x = Leaf(Tape(), x_)
        y = float(x)
        @test y isa Branch{Float64}
        @test unbox(y) == 4.0

        # Arrays
        X_ = [1,2,3,4]
        X = Leaf(Tape(), X_)
        Y = float(X)
        @test Y isa Branch{Vector{Float64}}
        @test unbox(Y) == Float64[1,2,3,4]

        # In expressions
        @test ∇(x->2x)(1) === (2,)
        @test ∇(x->2*float(x))(1) === (2.0,)
    end
end
