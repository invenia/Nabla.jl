@testset "finite_differencing" begin

    function check_discrepancy(g, ϵ_abs=1e-4, ϵ_rel=1e-3)
        # Create (extremely) simple problem.
        x0 = 10 * randn(5)
        x0 = 10.0
        δ_abs, δ_rel = discrepancy(g, (x0,), 1e-5)
        return all(map(all, map((x,y)->broadcast(<, x, y), δ_abs, ϵ_abs))) &&
               all(map(all, map((x,y)->broadcast(<, x, y), δ_rel, ϵ_rel)))
    end
    @test check_discrepancy(cos)
end
