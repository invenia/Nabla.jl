# @testset "sensitivities/functional" begin

    println(typeof(mapreduce))
    println(mapreduce(identity, +, randn(5)))
    s = mapreduce(abs2, +, Root([1, 2, 3, 4, 5], Tape()))
    ds = ∇(s)
    println(s)
    println(ds)

    # map(abs2, randn(5), randn(5))
# end
