print("core.jl... ")

# Check that the appropriate zero and one elements are returned by the scalar method.
function check_scalar()
    return getzero(5.0) == 0.0 && getone(4.933) == 1.0
end
@test check_scalar()

function check_vector()
    return getzero(randn(5)) == zeros(5) && getone(randn(5)) == ones(5)
end
@test check_vector()

function check_tuple()
    t = (5.0, (5.0, randn(2)))
    return getzero(t) == (0.0, (0.0, zeros(2))) && getone(t) == (1.0, (1.0, ones(2)))
end
@test check_tuple()


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
# @test check_basics_sum()

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
# @test check_basics_sumabs2()

println("passing.")
