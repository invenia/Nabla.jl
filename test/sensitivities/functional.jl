# @testset "sensitivities/functional" begin

    println(mapreduce(identity, +, randn(5)))
    mapreduce(abs2, +, Root(randn(5), Tape()))

    map(abs2, randn(5), randn(5))
# end
