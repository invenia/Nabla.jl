@testset "sensitivities/linalg/strided" begin

    let ϵ_abs = 1e-3, ϵ_rel = 1e-2, δ = 1e-6, N = 5

        # Test strided matrix-matrix multiplication sensitivities.
        for (f, tCA, tDA, CA, DA, tCB, tDB, CB, DB) in DiffBase.strided_matmul

            # Test allocating sensitivities.
            A, B, tape = randn(N, N), randn(N, N), Tape()
            A_, B_ = Leaf(A, tape), Leaf(B, tape)
            δ_abs, δ_rel = @eval discrepancy($f, ($A, $B), $δ)
            @test eval(DiffBase, f)(A, B) == eval(DiffBase, f)(A_, B_).val
            @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
            @test all(δ_abs[2] .< ϵ_abs) && all(δ_rel[2] .< ϵ_rel)

            # Test non-allocating sensitivities.
            g(A, B) = @eval $f($f($A, $f($A, $B)), $B)
            δ_abs, δ_rel = discrepancy(g, (A, B), δ)
            @test g(A, B) == g(A_, B_).val
            @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
            @test all(δ_abs[2] .< ϵ_abs) && all(δ_rel[2] .< ϵ_rel)
        end

        # # Test matrix-vector multiplication sensitivities.
        # for (f, tdA, CA, dA, tCb) in Nabla.strided_matvecmul

        #     # Test allocating sensitivities.
        #     A, b = randn(N, N), randn(N)
        #     δ_abs, δ_rel = @eval discrepancy($f, ($A, $b), $δ)
        #     @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
        #     @test all(δ_abs[2] .< ϵ_abs) && all(δ_rel[2] .< ϵ_rel)

        #     # Test non-allocating sensitivities.
        #     g(A, b) = @eval $f($A, $f($A, $b)) .+ $f($A, $b)
        #     δ_abs, δ_rel = discrepancy(g, (A, b), δ)
        #     @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
        #     @test all(δ_abs[2] .< ϵ_abs) && all(δ_rel[2] .< ϵ_rel)
        # end

        # # Test solving sensitivities.
        # for (f, C, tA, tB, arg1, arg2) in Nabla.strided_ldiv

        #     # Test allocating sensitivities.
        #     A, B = randn(5, 5), randn(5, 5)
        #     δ_abs, δ_rel = @eval discrepancy($f, ($A, $B), $δ)
        #     @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
        #     @test all(δ_abs[2] .< ϵ_abs) && all(δ_rel[2] .< ϵ_rel)
        # end
    end
end
