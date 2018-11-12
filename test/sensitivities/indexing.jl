indexing_generate_ȳ(rng, x, idcs...) = randn(rng, size(getindex(x, idcs...)))
indexing_generate_ȳ(rng, x, idcs::Int...) = randn(rng)

@testset "Indexing" begin

    # Check Vectors.
    let
        rng, P = MersenneTwister(123456), 10
        x, v = randn(rng, P), randn(rng, P)
        idcs = [1, P, P - 1, eachindex(x), 1:3:P, reverse(eachindex(x)), [P, 1]]
        for idc in idcs
            @test check_errs(x->getindex(x, idc), indexing_generate_ȳ(rng, x, idc), x, v)
        end
    end

    # Check Matrices.
    let
        rng, P, Q = MersenneTwister(123456), 10, 11
        x, v = randn(rng, P, Q), randn(rng, P, Q)

        # Linear indexing.
        idcs = [1, P * Q, P - 1, eachindex(x), 1:3:length(x), reverse(eachindex(x)), [P, 1]]
        for idc in idcs
            @test check_errs(x->getindex(x, idc), indexing_generate_ȳ(rng, x, idc), x, v)
        end

        # Cartesian indexing.
        idcs = [(1, 1), (P, Q), (P - 1, Q - 1), (1:P, 1:Q), (1:3:P, 1:4:Q),]
        for idc in idcs
            ȳ = indexing_generate_ȳ(rng, x, idc...)
            @test_broken check_errs(x->getindex(x, idc...), ȳ, x, v)
            @test check_errs(x->getindex(x, idc[1], idc[2]), ȳ, x, v)
        end
    end

    # Check dictionary using a slightly funny looking identity function.
    let
        rng, P = MersenneTwister(123456), 10
        ȳ, x, v = randn(rng, P), randn(rng, P), randn(rng, P)
        @test_broken check_errs(x->Dict('a'=>x)['a'], ȳ, x, v)
    end
end
