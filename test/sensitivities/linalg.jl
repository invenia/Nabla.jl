print("sensitivities/linalg.jl... ")

let ϵ_abs = 1e-3, ϵ_rel = 1e-2, δ = 1e-6, N = 5
    for (f, Ā, B̄) in vcat(AutoGrad2.matmul, AutoGrad2.ldiv, AutoGrad2.rdiv)
        A, B = randn(N, N), randn(N, N)
        δ_abs, δ_rel = @eval discrepancy($f, ($A, $B), $δ)
        @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel) &&
              all(δ_abs[2] .< ϵ_abs) && all(δ_rel[2] .< ϵ_rel)
    end
end

println("passing.")
