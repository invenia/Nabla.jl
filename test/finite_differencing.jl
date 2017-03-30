print("finite_differencing.jl... ")

# Note that this test relies on the gradients of sum and sumabs2 being implemented
# correctly in AGL. This is a fairly simple thing to ensure I guess...
function check_discrepancy(g, ϵ_abs=1e-4, ϵ_rel=1e-3)

    # Create (extremely) simple problem.
    f(x) = g(x)
    x0 = 10 * randn(5)
    δ_abs, δ_rel = discrepancy(f, (x0,), 1e-5)
    return all(map(all, map(.<, δ_abs, ϵ_abs))) && all(map(all, map(.<, δ_rel, ϵ_rel)))
end
@test check_discrepancy(sum)
@test check_discrepancy(sumabs2)

println("passing.")
