print("core.jl... ")

# @primitive sum{T<:AbstractArray}(x::T) y ȳ ȳ * ones(x)
# @primitive sumabs2{T<:AbstractArray}(x::T) y ȳ 2 * ȳ * x

# Test the core functionality of the package manually.
function check_basics_sum()

    # Define (very) simply function and it's gradient.
    f(x) = sum(x)
    df_manual(x) = ones(x)

    # Perform computation.
    x = Root(randn(5))
    y = f(x)
    grad(y)

    # Compare hand-coded with AD.
    return all(df_manual(x.val) == x.dval)
end
@test check_basics_sum()

# Test the core functionality of the package manually.
function check_basics_sumabs2()

    # Define (very) simply function and it's gradient.
    f(x) = sumabs2(x)
    df_manual(x) = 2 * x

    # Perform computation.
    x = Root(randn(5))
    y = f(x)
    grad(y)

    # Compare hand-coded with AD.
    return all(df_manual(x.val) == x.dval)
end
@test check_basics_sumabs2()

println("passing.")
