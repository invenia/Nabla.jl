import Base: sum, sumabs2

# Define forward-pass expansion.
sum{T<:Node}(x::T) = Branch{Float64}(sum, (x,))

# Define reverse-mode sensitivity. Pretend I'm not commiting function piracy for now. Will
# fix this later.
sum(z̄::Float64, z::Float64, x) = (z̄ * ones(x),)

sumabs2{T<:Node}(x::T) = Branch{Float64}(sumabs2, (x,))
sumabs2(z̄::Float64, z::Float64, x) = (2 * z̄ * x,)

# Test the core functionality of the package manually.
function check_basics_sum()

    # Define (very) simply function and it's gradient.
    f(x) = sum(x)
    df_manual(x) = ones(x)

    # Perform computation.
    x = Root{Vector{Float64}}(randn(5))
    y = f(x)
    dump(y)
    grad(y)

    # Compare hand-coded with AD.
    println("df_manual = ", df_manual(x.val))
    println("df_auto = ", x.dval)
    return all(df_manual(x.val) == x.dval)
end
@test check_basics_sum()

# Test the core functionality of the package manually.
function check_basics_sumabs2()

    # Define (very) simply function and it's gradient.
    f(x) = sumabs2(x)
    df_manual(x) = 2 * x

    # Perform computation.
    x = Root{Vector{Float64}}(randn(5))
    y = f(x)
    dump(y)
    grad(y)

    # Compare hand-coded with AD.
    println("df_manual = ", df_manual(x.val))
    println("df_auto = ", x.dval)
    return all(df_manual(x.val) == x.dval)
end
@test check_basics_sumabs2()
