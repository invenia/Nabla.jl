print("sensitivities/array.jl... ")

let N = 4, ϵ_abs = 1e-5, ϵ_rel = 1e-4, δ = 1e-6

    # Helper functions to for correctness checking.
    check_abs(x) = all(x .< ϵ_abs)
    check_rel(x) = all(x .< ϵ_rel)

    # # Check sensitivities for elementwise application of functions of a single argument.
    # for (f, x̄, range) in AutoGrad2.unary_sensitivities
    #     x = rand(N) * (range[2] - range[1]) + range[1]

    #     δ_abs, δ_rel = discrepancy(eval(f), (x,), δ, [true])
    #     @test all(map(check_abs, δ_abs)) && all(map(check_rel, δ_rel))
    # end

    # # Check sensitivities for elementwise application of functions of two arguments.
    # for (f, new_x̄, new_ȳ, update_x̄, update_ȳ, range_x, range_y) in AutoGrad2.binary_sensitivities_elementwise
    #     x = rand(N) * (range_x[2] - range_x[1]) - range_x[1]
    #     y = rand(N) * (range_y[2] - range_y[1]) - range_y[1]

    #     δ_abs, δ_rel = discrepancy(eval(f), (x, y), δ, [true, true])
    #     @test all(map(check_abs, δ_abs)) && all(map(check_rel, δ_rel))

    #     # Testing with previously allocated memory.
    #     g = :((x, y)->$f($f($f(x, y), y), x))
    #     δ_abs, δ_rel = discrepancy(eval(g), (x, y), δ, [true, true])
    #     @test all(map(check_abs, δ_abs)) && all(map(check_rel, δ_rel))

    #     # Testing with different sized arrays.
    #     x = randn(N, 4, 1) * (range_x[2] - range_x[1]) - range_x[1]
    #     y = randn(1, 1, 6) * (range_y[2] - range_y[1]) - range_y[1]
    #     δ_abs, δ_rel = discrepancy(eval(f), (x, y), δ, [true, true])
    #     @test all(map(check_abs, δ_abs)) && all(map(check_rel, δ_rel))
    # end

    # Test sensitivities for reduce functions of a single argument.
    M, P = 2, 3
    for (f, x̄) in AutoGrad2.reduce
        # Generate some random dense arrays.
        x1, x2, x3 = randn(N), randn(N, M), randn(N, M, P)
        x1[1], x2[1], x3[1] = x1[1] + 100.0, x2[1] + 100.0, x3[1] + 100.0
        x1[end], x2[end], x3[end] = x1[end] - 100.0, x2[end] - 100.0, x3[end] - 100.0

        # Compute discrepancies for each array.
        δ_abs_1, δ_rel_1 = @eval discrepancy($f, ($x1,), $δ)
        δ_abs_2, δ_rel_2 = @eval discrepancy($f, ($x2,), $δ)
        δ_abs_3, δ_rel_3 = @eval discrepancy($f, ($x3,), $δ)
        δ_abs_4, δ_rel_4 = @eval discrepancy($f, ($x1, 1), $δ, [true, false])
        δ_abs_5, δ_rel_5 = @eval discrepancy($f, ($x2, 1), $δ, [true, false])
        δ_abs_6, δ_rel_6 = @eval discrepancy($f, ($x2, 1), $δ, [true, false])
        δ_abs_7, δ_rel_7 = @eval discrepancy($f, ($x3, [1, 3]), $δ, [true, false])

        # Check that we are within tolerance for everything.
        @test all(map(check_abs, δ_abs_1)) && all(map(check_rel, δ_rel_1)) &&
              all(map(check_abs, δ_abs_2)) && all(map(check_rel, δ_rel_2)) &&
              all(map(check_abs, δ_abs_3)) && all(map(check_rel, δ_rel_3)) &&
              all(map(check_abs, δ_abs_4)) && all(map(check_rel, δ_rel_4)) &&
              all(map(check_abs, δ_abs_5)) && all(map(check_rel, δ_rel_5)) &&
              all(map(check_abs, δ_abs_6)) && all(map(check_rel, δ_rel_6)) &&
              all(map(check_abs, δ_abs_7)) && all(map(check_rel, δ_rel_7))

        # # Scalar test case.
        # x = randn()
        # δ_abs, δ_rel = @eval discrepancy($f, ($x,), $δ)
        # @test all(map(check_abs, δ_abs)) && all(map(check_abs, δ_rel))
    end
end

println("passing.")
