@testset "sensitivities/scalar" begin
    let ϵ_abs = 1e-5, ϵ_rel = 1e-4, δ = 1e-6

        function unary_test(f, x)
            δ_abs, δ_rel = discrepancy(f, (x,), δ)
            (any(δ_abs .> ϵ_abs) || any(δ_rel .> ϵ_rel)) && println((f, δ_abs, δ_rel, x))
            @test all(δ_abs .< ϵ_abs) && all(δ_rel .< ϵ_rel)
        end

        for (f, x̄, range) in AutoGrad2.unary_sensitivities
            for _ in 1:10
                unary_test(f, rand() * (range[2] - range[1]) + range[1])
            end
            unary_test(f, range[1])
            unary_test(f, range[2])
        end

        function binary_test(f, x, y)
            δ_abs, δ_rel = discrepancy(f, (x, y), δ)
            (any(δ_abs .> ϵ_abs) || any(δ_rel .> ϵ_rel)) && println((f, δ_abs, δ_rel, x, y))
            @test all(δ_abs .< ϵ_abs) && all(δ_rel .< ϵ_rel)
        end

        for (f, x̄, ȳ, range_x, range_y) in AutoGrad2.binary_sensitivities
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
    end

end
