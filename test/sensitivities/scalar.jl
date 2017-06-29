@testset "sensitivities/scalar" begin

    let ϵ_abs = 1e-5, ϵ_rel = 1e-4, δ = 1e-6
        for (f, x̄, range) in AutoGrad2.unary_sensitivities
            x = rand() * (range[2] - range[1]) + range[1]
            δ_abs, δ_rel = discrepancy(f, (x,), δ)
            (any(δ_abs .> ϵ_abs) || any(δ_rel .> ϵ_rel)) && println((f, δ_abs, δ_rel))
            @test all(δ_abs .< ϵ_abs) && all(δ_rel .< ϵ_rel)
        end
        for (f, x̄, ȳ, range_x, range_y) in AutoGrad2.binary_sensitivities
            x = rand() * (range_x[2] - range_x[1]) - range_x[1]
            y = rand() * (range_x[2] - range_x[1]) - range_y[1]
            δ_abs, δ_rel = discrepancy(f, (x, y), δ)
            @test all(δ_abs .< ϵ_abs) && all(δ_rel .< ϵ_rel)
        end
    end

end
