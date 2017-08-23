@testset "sensitivities/scalar" begin

    let ϵ_abs = 1e-5, ϵ_rel = 1e-4, v = 1e-6, ȳ = 5.0, z̄ = 4.0

        let
            @test ∇(identity, Arg{1}, 5.0, 4.0, 3.0, 2.0) == 3.0
            @test ∇(identity, Arg{1}, 5) == 1
            @test ∇(identity, Arg{1}, 5.0) == 1.0
        end

        unary_check(f, x) = check_errs(eval(f), ȳ, x, v, ϵ_abs, ϵ_rel)
        for (f, x̄, range) in Nabla.unary_sensitivities
            for _ in 1:10
                @test unary_check(f, rand() * (range[2] - range[1]) + range[1])
            end
            @test unary_check(f, range[1])
            @test unary_check(f, range[2])
        end

        function binary_test(f, x, y)

            # Construct two lambdas, one for each argument.
            f = eval(f)
            λx, λy = (x)->f(x, y), (y)->f(x, y)

            # Compute error w.r.t. first argument and check results.
            δ_abs_x, δ_rel_x = check_Dv(λx, z̄, x, v)
            (δ_abs_x > ϵ_abs || δ_rel_x > ϵ_rel || isnan(δ_abs_x) || isnan(δ_rel_x)) &&
                print_tol_err(f, z̄, x, v, δ_abs_x, δ_rel_x)
            @test δ_abs_x < ϵ_abs && δ_rel_x < ϵ_rel

            # Compute error w.r.t. second argument and check results.
            δ_abs_y, δ_rel_y = check_Dv(λy, z̄, y, v)
            (δ_abs_y > ϵ_abs || δ_rel_y > ϵ_rel || isnan(δ_abs_y) || isnan(δ_rel_y)) &&
                print_tol_err(f, z̄, y, v, δ_abs_y, δ_rel_y)
            @test δ_abs_y < ϵ_abs && δ_rel_y < ϵ_rel
        end

        for (f, x̄, ȳ, range_x, range_y) in Nabla.binary_sensitivities
            for _ in 1:10
                x = rand() * (range_x[2] - range_x[1]) + range_x[1]
                y = rand() * (range_y[2] - range_y[1]) + range_y[1]
                binary_test(f, x, y)
            end
            binary_test(f, range_x[1], range_y[1])
            binary_test(f, range_x[1], range_y[2])
            binary_test(f, range_x[2], range_y[1])
            binary_test(f, range_x[2], range_y[2])
        end

        # Test exponentiation amibiguity is resolved.
        @test ∇(x->x^2)(1) == (2.0,)
    end
end
