@testset "Array" begin
    @test size(Leaf(Tape(), ones(1, 2, 3, 4))) == (1, 2, 3, 4)
    @test length(Leaf(Tape(), 1:3)) == 3

    let rng = MersenneTwister(123456)
        x = randn(2, 10)
        f1 = x̂ -> reshape(x̂, 5, 4)
        f2 = x̂ -> reshape(x̂, (5, 4))
        @test check_errs(f1, f1(x), x, randn(size(x)...))
        @test check_errs(f2, f2(x), x, randn(size(x)...))
    end

    a = rand(3, 2); b = rand(3); c = rand(3, 3);
    f(a, b, c) = sum(hcat(2*a, 3*b, 4*c))
    @test ∇(f)(a,b,c) == (2*ones(3, 2), 3*ones(3), 4*ones(3, 3))

    a = rand(2, 4); b = rand(1, 4); c = rand(3, 4);
    g(a, b, c) = sum(vcat(2*a, 3*b, 4*c))
    @test ∇(g)(a,b,c) == (2*ones(2, 4), 3*ones(1, 4), 4*ones(3, 4))

    @test check_errs(x->fill(x, 4, 4), randn(4, 4), randn(), randn())
    @test check_errs(x->fill(x, (4, 4)), randn(4, 4), randn(), randn())
    x = Leaf(Tape(), 6)
    y = fill(x, 3)
    @test y isa Branch{Vector{Int}}
    @test getfield(y, :f) === Base.fill
end
