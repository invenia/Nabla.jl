export discrepancy

"""
Compare numerical estimate of projection of gradient with the actual projection of the
gradient, computed via AGL.

Inputs:
f  - function to compute discrepancy w.r.t. Must be a primitive.
x0 - point at which to consider discrepancy.
δ - perturbation to apply to each element in turn.
diff (optional) - a vector indicating which of the arguments are differentiable.
trans (optional) - transformation of the output of f to ensure that output is deterministic.

Returns:
δ_abs - absolute error.
δ_rel - relative error.
"""
function discrepancy(f::Function, x0::Tuple, δ::Float64, diff::Vector=[], trans::Function=x->x)

    println("f is ", f)

    # If diff doesn't contain anything, then differentiate all arguments.
    diff = diff == [] ? [true for j in x0] : diff

    δ_abs, δ_rel = [], []
    for n in eachindex(x0)

        # If argument is differentiable then check correctness via finite differencing. If
        # argument is not differentiable then just return 0.
        if diff[n] == true

            # Compute x̄ using AutoDiff.
            x = collect(Any, x0)
            x[n] = Root(x0[n])
            y = f(x...)
            grad(y)
            println("dval is ", x[n].dval)

            # Estimate x̄ using finite differencing.
            x̄ = estimate_x̄(f, x0, δ, x0[n], n, trans)
            println("x̄ is ", x̄)

            # Compute absolute and relative errors for this argument.
            push!(δ_abs, abs(x̄ - x[n].dval))
            push!(δ_rel, δ_abs[n] ./ abs(x̄ + 1e-15))
        else
            push!(δ_abs, 0.0)
            push!(δ_rel, 0.0)
        end
    end
    return δ_abs, δ_rel
end

function estimate_x̄(f::Function, x0::Tuple, δ::Float64, x0n::Float64, n::Int, trans::Function)
    x1, x2 = collect(deepcopy(x0)), collect(deepcopy(x0))
    x1[n], x2[n] = x0[n] + δ, x0[n] - δ
    return sum(map(-, trans(f(x1...)), trans(f(x2...))) ./ 2δ)
end

function estimate_x̄(f::Function, x0::Tuple, δ::Float64, x0n::AbstractArray, n::Int, trans::Function)
    x̄ = zeros(x0n)
    for j in eachindex(x0n)
        x1, x2 = deepcopy(x0), deepcopy(x0)
        x1[n][j], x2[n][j] = x0n[j] + δ, x0n[j] - δ
        x̄[j] = sum(map(-, trans(f(x1...)), trans(f(x2...))) ./ 2δ)
    end
    return x̄
end
