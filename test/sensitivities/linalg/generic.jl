@testset "sensitivities/linalg/generic" begin

    let ϵ_abs = 1e-3, ϵ_rel = 1e-2, δ = 1e-5, N = 2

        # Test unary linalg sensitivities.
        for (f, T_In, T_Out, X̄, bounds) in DiffBase.unary_linalg_optimisations

            # Test allocating sensitivities.
            Z = rand(Uniform(bounds[1], bounds[2]), N, N)
            X = Z'Z + UniformScaling(1e-3)
            X_ = Leaf(Tape(), X)
            δ_abs, δ_rel = @eval discrepancy($f, ($X,), $δ)

            # Check that the correct output is computed.
            @test eval(DiffBase, f)(X) == eval(DiffBase, f)(X_).val

            # Check that sensitivities w.r.t. the input are correct.
            (any(δ_abs[1] .> ϵ_abs) || any(δ_rel[1] .> ϵ_rel)) &&
                println((f, δ_abs[1], δ_rel[1], X))
            @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)
        end

        # Test binary linalg optimisations.
        for (f, T_A, T_B, T_Y, Ā, B̄, errs) in DiffBase.binary_linalg_optimisations

            # Test allocating sensitivities.
            Z1, Z2 = randn(N, N), randn(N, N)
            # Z1, Z2 = diagm(randn(N)), diagm(randn(N))
            # Z1, Z2 = rand(Uniform(1.0, 1.5), N, N), rand(Uniform(1.0, 1.5), N, N)
            A, B, tape = Z1'Z1 + UniformScaling(1.0), Z2'Z2 + UniformScaling(1.0), Tape()
            A, B, tape = Z1, Z2, Tape()
            A_, B_ = Leaf(tape, A), Leaf(tape, B)
            ∇f = ∇(eval(DiffBase, f)(A_, B_))

            δ_abs, δ_rel = errs(eval(DiffBase, f), A_, B_, ∇f[A_], ∇f[B_])

            # Check that the correct output is computed.
            @test eval(DiffBase, f)(A, B) == eval(DiffBase, f)(A_, B_).val

            # Check that sensitivities w.r.t. the first input are correct.
            (any(δ_abs[1] .> ϵ_abs) || any(δ_rel[1] .> ϵ_rel)) &&
                println(("1", f, δ_abs[1], δ_rel[1], A, B, ∇f[A_], ∇f[B_]))
            @test all(δ_abs[1] .< ϵ_abs) && all(δ_rel[1] .< ϵ_rel)

            # Check that sensitivities w.r.t. the second input are correct.
            (any(δ_abs[2] .> ϵ_abs) || any(δ_rel[2] .> ϵ_rel)) &&
                println(("2", f, δ_abs[2], δ_rel[2], A, B, ∇f[A_], ∇f[B_]))
            @test all(δ_abs[2] .< ϵ_abs) && all(δ_rel[2] .< ϵ_rel)
        end
    end
end
