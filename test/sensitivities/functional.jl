@testset "sensitivities/functional" begin

    println(mapreduce(identity, +, randn(5)))
    mapreduce(abs, +, Root(randn(5), Tape()))

end
