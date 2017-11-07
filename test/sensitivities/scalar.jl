using DiffRules: diffrule

@testset "Scalar domains" begin
    @test in_domain(sin, 10.)
    @test in_domain(cos, 10.)
    @test !in_domain(acos, 10.)
    @test !in_domain(asin, 10.)
    @test get(domain1(sin)) == (minimum(points), maximum(points))
    @test get(domain1(log)) == (minimum(points[points .> 0]), maximum(points))
    @test get(domain1(acos)) == (minimum(points[points .> -1]),
                                 maximum(points[points .< 1]))
    @test get(domain2((+))) == ((minimum(points), maximum(points)),
                                (minimum(points), maximum(points)))
    @test get(domain2((^))) == ((minimum(points[points .> 0]), maximum(points)),
                                (minimum(points), maximum(points)))
    @test get(domain2(beta)) == ((minimum(points[points .> 0]), maximum(points)),
                                 (minimum(points[points .> 0]), maximum(points)))
end

@testset "Scalar" begin
    let v = 1.0, ȳ = 5.0, z̄ = 4.0, rng = MersenneTwister(123456)
        let
            @test ∇(identity, Arg{1}, 5.0, 4.0, 3.0, 2.0) == 3.0
            @test ∇(identity, Arg{1}, 5) == 1
            @test ∇(identity, Arg{1}, 5.0) == 1.0
        end

        unary_check(f, x) = check_errs(eval(f), ȳ, x, v)
        for (package, f) in Nabla.unary_sensitivities
            domain = domain1(eval(f))
            isnull(domain) && error("Could not determine domain for $f.")
            lb, ub = get(domain)
            randx = () -> rand(rng) * (ub - lb) + lb

            for _ in 1:10
                @test unary_check(f, randx())
            end
        end

        for (package, f) in Nabla.binary_sensitivities
            ∂f∂x, ∂f∂y = diffrule(package, f, :x, :y)

            if ∂f∂x == :NaN && ∂f∂y != :NaN
                # Assume that the first argument is integer-valued.
                domain = domain1(y -> eval(f)(0, y))
                isnull(domain) && error("Could not determine domain for $f.")
                lb, ub = get(domain)
                randx = () -> rand(rng, 0:5)
                randy = () -> rand(rng) * (ub - lb) + lb

                for _ in 1:10
                    x = randx()
                    @test check_errs(y -> eval(f)(x, y), ȳ, randy(), v)
                end
            elseif ∂f∂x != :NaN && ∂f∂y == :NaN
                # Assume that the second argument is integer-valued.
                domain = domain1(x -> eval(f)(x, 0))
                isnull(domain) && error("Could not determine domain for $f.")
                lb, ub = get(domain)
                randx = () -> rand(rng) * (ub - lb) + lb
                randy = () -> rand(rng, 0:5)

                for _ in 1:10
                    y = randy()
                    @test check_errs(x -> eval(f)(x, y), randx(), ȳ, v)
                end
            elseif ∂f∂x != :NaN && ∂f∂y != :NaN
                domain = domain2(eval(f))
                isnull(domain) && error("Could not determine domain for $f.")
                (x_lb, x_ub), (y_lb, y_ub) = get(domain)
                randx = () -> rand(rng) * (x_ub - x_lb) + x_lb
                randy = () -> rand(rng) * (y_ub - y_lb) + y_lb

                for _ in 1:10
                    @test check_errs(eval(f), z̄, (randx(), randy()), (v, v))
                end
            else
                error("Cannot test $f: $f is not differentiable in either argument.")
            end
        end

        # Test whether the exponentiation amibiguity is resolved.
        @test ∇(x -> x^2)(1) == (2.0,)
    end
end
