print("composite/simple.jl... ")

using BenchmarkTools

# Tests for simple of functions.
let ϵ_abs = 1e-5, ϵ_rel = 1e-4, δ = 1e-6

    # Simple helper functions to iterate over tuple-valued arguments.
    check_abs(x) = all(x .< ϵ_abs)
    check_rel(x) = all(x .< ϵ_rel)

    # Logistic function.
    let x = randn(10, 10), f(x) = 1 ./ (1 .+ exp(-x))

        # Check via finite differencing.
        δ_abs, δ_rel = discrepancy(f, (x,), δ, [true])
        @test all(map(check_abs, δ_abs)) && all(map(check_rel, δ_rel))

        # Check via hand-coded gradients (because why not?)
        ∇x = f(x) .* (1 .- f(x))
        xn = Root(x, Tape())
        df = ∇(f(xn))
        @test maximum(abs(∇x - df[xn])) < ϵ_abs
    end

    # 2D Rosenbrock function.
    let x = randn(25), y = randn(25), f(x, y) = (1.0 .- x) .* (1.0 .- x) .+ 100.0 .* (y .- x .* x) .* (y .- x .* x)
        δ_abs, δ_rel = discrepancy(f, (x, y), δ, [true, true])
        @test all(map(check_abs, δ_abs)) && all(map(check_rel, δ_rel))
    end

    # Simple over-writing of variables.
    let
        ftape = Tape()
        xr = Root(randn(5), ftape)
        yr = Root(randn(5), ftape)
        z = xr .* yr
        z = xr .+ yr
        rtape = ∇(xr)
        @test all(rtape[xr] == 1.0) && all(rtape[yr] == 1.0) && !isdefined(rtape, 3)
    end
end

println("passing.")
