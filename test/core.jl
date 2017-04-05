print("core.jl... ")

# Test the core functionality of the package manually.
function check_basics_sum()

    # Define (very) simply function and it's gradient.
    f(x) = sum(x)
    df_manual(x) = ones(x)

    # Perform computation.
    x = Root(randn(5), Tape())
    y = f(x)
    rvs_tape = ∇(y)

    # Compare hand-coded with AD.
    return all(df_manual(x.val) == rvs_tape[x])
end
@test check_basics_sum()

# Test the core functionality of the package manually.
function check_basics_sumabs2()

    # Define (very) simply function and it's gradient.
    f(x) = sumabs2(x)
    df_manual(x) = 2 * x

    # Perform computation.
    x = Root(randn(5), Tape())
    y = f(x)
    rvs_tape = ∇(y)

    # Compare hand-coded with AD.
    return all(df_manual(x.val) == rvs_tape[x])
end
@test check_basics_sumabs2()

println("passing.")
