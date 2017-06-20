print("primitive.jl... ")

# import Base.sum

# out = sensitivity(:(sum{T<:Union{AbstractArray, Real}}(x::T)),
#     (:x̄, :(x̄ = broadcast!(x->x, similar(x), ȳ)), :(broadcast!(+, x̄, x̄, ȳ))), :y, :ȳ)
# println(out)
# eval(out)

function check_sensitivity_func()
    x_ = randn(5)
    x = Root(x_, Tape())
    y = sum(x)
    println(y)
    ∇y = ∇(y)
    return ∇y[x] == ones(x_)
end
@test check_sensitivity_func()

println("passing.")
