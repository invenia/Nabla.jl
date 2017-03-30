print("sensitivities/array.jl... ")

let N = 4, ϵ_abs = 1e-5, ϵ_rel = 1e-4, δ = 1e-6

    # Check sensitivities for elementwise application of functions of a single argument.
    for (f, x̄, range) in AutoGrad2.unary_sensitivities
        x = rand(N) * (range[2] - range[1]) + range[1]

        # Compute sensitivities using elementwise implementation.
        x_node = Root(x)
        @eval grad($f($x_node))

        # Explicitely compute sensitivities for each element.
        dx_explicit = Vector{Float64}(N)
        for n in 1:N
            x_n = Root(x[n])
            @eval grad($f($x_n))
            dx_explicit[n] = x_n.dval
        end

        @test all(dx_explicit .== x_node.dval)
    end

    # Check sensitivities for elementwise application of functions of two arguments.
    for (f, x̄, ȳ, range_x, range_y) in AutoGrad2.binary_sensitivities_elementwise
        x = rand(N) * (range_x[2] - range_x[1]) - range_x[1]
        y = rand(N) * (range_y[2] - range_y[1]) - range_y[1]

        # Compute sensitivities using elementwise implementation.
        x_node, y_node = Root(x), Root(y)
        @eval grad($f($x_node, $y_node))

        # Explicitly compute sensitivities for each element.
        dx_explicit, dy_explicit = Vector{Float64}(N), Vector{Float64}(N)
        for n in 1:N
            x_n, y_n = Root(x[n]), Root(y[n])
            @eval grad($f($x_n, $y_n))
            dx_explicit[n], dy_explicit[n] = x_n.dval, y_n.dval
        end

        @test all(dx_explicit .== x_node.dval) && all(dy_explicit .== y_node.dval)
    end

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
        @test all(δ_abs_1[1] .< ϵ_abs) && all(δ_rel_1[1] .< ϵ_rel) &&
              all(δ_abs_2[1] .< ϵ_abs) && all(δ_rel_2[1] .< ϵ_rel) &&
              all(δ_abs_3[1] .< ϵ_abs) && all(δ_rel_3[1] .< ϵ_rel) &&
              all(δ_abs_4[1] .< ϵ_abs) && all(δ_rel_4[1] .< ϵ_rel) &&
              all(δ_abs_5[1] .< ϵ_abs) && all(δ_rel_5[1] .< ϵ_rel) &&
              all(δ_abs_6[1] .< ϵ_abs) && all(δ_rel_6[1] .< ϵ_rel) &&
              all(δ_abs_7[1] .< ϵ_abs) && all(δ_rel_7[1] .< ϵ_rel)
    end
end

println("passing.")
