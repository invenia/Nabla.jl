print("primitive.jl... ")

foo(x::Real, y::Float64) = println("Aha!")
out = sensitivity(
    :(foo{T<:AbstractFloat}(x::Real, y::T)),
    Vector{Tuple}([
        (:x̄, :(x̄ = x), :(x̄ = x)),
        (:nothing,)]),
        # (:ȳ, :(ȳ = y), :(ȳ = y))]),
    :z,
    :z̄,
    :(println("Some preprocessing.")))
eval(out[1])
eval(out[2])

function check_sensitivity_func()
    println()
    println(out[1])
    println()
    println(out[2])
    println(methods(foo))
    println(foo(5, 4.0))
    println(foo(Root(5, Tape()), 4.0))
    return true
end
@test check_sensitivity_func()

println("passing.")
