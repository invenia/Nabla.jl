print("util.jl... ")

function check_typedargs(types)
    args = [:x, :y]
    expected = [Expr(:(::), :x, :T), Expr(:(::), :y, :(V <: AbstractFloat))]
    actual = AutoGrad2.typedargs(args, types)
    return expected == actual
end
@test check_typedargs([:T, :(V <: AbstractFloat)])
@test_throws ArgumentError check_typedargs([:T, :(V <: AbstractFloat), 5.0])

println("passing.")
