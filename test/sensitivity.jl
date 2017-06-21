print("sensitivity.jl... ")

function check_sensitivity_func()
    x_ = randn(5)
    x = Root(x_, Tape())
    y = sum(x)
    ∇y = ∇(y)
    return all(∇y[x] == ones(x_))
end
@test check_sensitivity_func()

println("passing.")
