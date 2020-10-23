@testset "core" begin

@testset "Tape" begin
    # Simple tests for `Tape`.
    @test getindex(setindex!(Tape(5), "hi", 5), 5) == "hi"
    @test lastindex(Tape(50)) == 50
    @test eachindex(Tape(50)) == Base.OneTo(50)
    @test length(Tape()) == 0
    @test length(Tape(50)) == 50
    @test !isassigned(Tape(1), 1)
    @test !isassigned(Tape(26), 26)

    # Check that printing works as expected.
    @testset "printing" begin
        buffer = IOBuffer()
        show(buffer, Tape())
        @test String(take!(buffer)) == "Tape with 0 elements"

        buffer = IOBuffer()
        show(buffer, Tape(1))
        @test String(take!(buffer)) == "Tape with 1 element:\n  [1]: #undef"

        buffer = IOBuffer()
        show(buffer, Tape(2))
        @test String(take!(buffer)) == "Tape with 2 elements:\n  [1]: #undef\n  [2]: #undef"

        buffer = IOBuffer()
        tape_ = Tape(1)
        tape_[1] = 5
        show(buffer, tape_)
        @test String(take!(buffer)) == "Tape with 1 element:\n  [1]: 5"
    end

    # Check isassigned consistency.
    leaf = Leaf(Tape(), 5)
    @test isassigned(tape(tape(leaf)), pos(leaf)) == isassigned(tape(leaf), leaf)

    # Simple tests for `Leaf`.
    tp1, tp2 = Tape(), Tape(50)
    @test tape(Leaf(tp1, 5.0)) == tp1
    @test unbox(Leaf(tp1, 5.0)) == 5.0
    @test pos(Leaf(tp1, 5.0)) == 3
    @test pos(Leaf(tp2, 5)) == 51
    @test unbox(Leaf(tp2, 5)) == 5
    @test tape(Leaf(tp2, 5)) == tp2

    # Simple tests for `Branch`.
    foo_coeff = 10
    foo(x::Real) = foo_coeff * x
    foo(x::Node{T} where T<:Real) = Branch(foo, (x,), tape(x))
    Nabla.∇(::typeof(foo), ::Type{Arg{1}}, p, y, ȳ, x) = ȳ * foo_coeff
    function get_new_branch()
        leaf = Leaf(Tape(), 5)
        return foo(leaf)
    end
    @test isa(get_new_branch(), Branch)
    @test unbox(get_new_branch()) == 50
    @test getfield(get_new_branch(), :f) == foo
    @test length(getfield(get_new_branch(), :args)) == 1
    @test isa(getfield(get_new_branch(), :args)[1], Leaf)
    @test unbox(getfield(get_new_branch(), :args)[1]) == 5
    @test pos(getfield(get_new_branch(), :args)[1]) == 1
    @test pos(get_new_branch()) == 2
    @test tape(getfield(get_new_branch(), :args)[1]) != tape(get_new_branch())
    branch = get_new_branch()
    @test tape(branch) == tape(getfield(branch, :args)[1])

    # Simple test for `pos`.
    @test Nabla.pos(1) == -1
    @test Nabla.pos("hi") == -1
    @test Nabla.pos(Leaf(Tape(), 5)) == 1
    @test Nabla.pos(Leaf(Tape(50), 5.0)) == 51

    # Simple tests for `unbox`.
    @test Nabla.unbox(1) == 1
    @test Nabla.unbox("hi") == "hi"
    @test Nabla.unbox(Leaf(Tape(), 5)) == 5
    @test Nabla.unbox(Leaf(Tape(), 5.0)) == 5.0

    # Simple tests for reverse_tape.
    @test isa(Nabla.reverse_tape(Leaf(Tape(), 5), 5), Tape)
    @test length(Nabla.reverse_tape(Leaf(Tape(), 5), 5)) == 1
    @test Nabla.reverse_tape(Leaf(Tape(), 5), 4)[end] == 4
    @test isassigned(Nabla.reverse_tape(Leaf(Tape(), 5), 5), 1)
    @test isassigned(Nabla.reverse_tape(get_new_branch(), 5), 2)
    @test !isassigned(Nabla.reverse_tape(get_new_branch(), 5), 1)

    # Simple tests for `propagate`.
    @test Nabla.propagate(Leaf(Tape(), 5), Tape()) == nothing
    br = get_new_branch()
    rvs_tape = Nabla.reverse_tape(br, 5)
    @test Nabla.propagate(br, rvs_tape) == nothing
    @test rvs_tape[br] == 5
    @test rvs_tape[br.args[1]] == foo_coeff * 5

    # Simple integration tests for ∇.
    br = get_new_branch()
    @test ∇(br, 3)[br] == 3
    @test ∇(br, 2)[br.args[1]] == 2 * foo_coeff

end # testset Tape


@testset "Check that functions involving `isapprox` can be differentiated" begin
    @testset "Test Case 1" begin
        f(x) = x ≈ 5.0 ? 1.0 : 3.0 * x
        g(x) = 5.0 * x
        h(x) = g(x) ≈ 25.0 ? x : f(x) + g(x)
        ∇f = ∇(f)
        ∇h = ∇(h)
        @test ∇f(5.0) == (0.0,)
        @test ∇f(6.0) == (3.0,)
        @test ∇h(5.0) == (1.0,)
        @test ∇h(6.0) == (8.0,)
    end

    @testset "Test Case 2" begin
        f(x) = x ≈ [5.0] ? 1.0 : 3.0 * sum(x)
        ∇f = ∇(f)
        @test ∇f([5.0]) == ([0.0],)
        @test ∇f([6.0]) == ([3.0],)
        f(x, y) = x ≈ y ? 2y : 3x
        ∇f = ∇(f)
        @test ∇f(5.0, 5.0) == (0.0, 2.0)
        @test ∇f(6.0, 5.0) == (3.0, 0.0)
    end
end

@testset "Check that functions with extra, unused variables can be differentiated" begin
    f(a,b,c,d) = a*c
    ∇f = ∇(f)
    g(a,b) = 12
    ∇g = ∇(g)

    @test ∇f(1,2,3,4) == (3, 0, 1, 0)
    @test ∇f(1,[2.0],3,4.0) == (3, [0.0], 1, 0.0)
    @test ∇g(1,2) == (0,0)
end

@testset "Check that functions with `zero` and `one` can be differentiated" begin
    f(a) = zero(a)
    g(a) = one(a)
    h(a) = zero(3 * a) + one(4 * a)
    ∇f = ∇(f)
    ∇g = ∇(g)
    ∇h = ∇(h)

    @test ∇f(1) == (0,)
    @test ∇f([1]) == ([0],)
    @test ∇g(4) == (0,)
    @test ∇h(8) == (0,)
end

@testset "Check that the convenience implementation of ∇ works as intended." begin
    f(x, y) = 2x + y
    ∇f = ∇(f)
    ∇f_out = ∇(f; get_output=true)

    @test_throws MethodError ∇f(randn(5), randn(5))
    x, y = randn(), randn()
    ∇z = ∇(f(Leaf.(Tape(), (x, y))...))
    @test ∇f(x, y) == (∇z[1], ∇z[2])
    z, (∇x, ∇y) = ∇f_out(x, y)
    @test unbox(z) == f(x, y)
    @test (∇x, ∇y) == ∇f(x, y)
end

@testset "get_output" begin
    y = ∇(-, get_output=true)(2)
    @test y isa Tuple{Branch{Int}, Tuple{Int}}
    @test last(y) == (-1,)
    @test ∇(unbox, get_output=true)(2) == (2, (0,))
end

@testset "Tests for zero'd and one'd containers." begin
    import Nabla: zerod_container, oned_container
    @test zerod_container(1.0) == 0.0
    @test zerod_container(1) == 0
    @test oned_container(1.0) == 1.0
    @test oned_container(5) == 1

    @test zerod_container(randn(5)) == zeros(5)
    @test oned_container(randn(5)) == ones(5)
    @test zerod_container(randn(5, 4, 3, 2, 1)) == zeros(5, 4, 3, 2, 1)
    @test oned_container(randn(5, 4, 3, 2, 1)) == ones(5, 4, 3, 2, 1)

    @test zerod_container((1.0, 1)) == (0.0, 0)
    @test oned_container((0, 0.0)) == (1, 1.0)
    @test zerod_container((randn(), randn(5))) == (0.0, zeros(5))
    @test oned_container((randn(5), randn())) == (ones(5), 1.0)
    @test zerod_container((1.0, (randn(5), randn(5)))) == (0.0, (zeros(5), zeros(5)))
    @test oned_container((randn(), (randn(5), randn(5)))) == (1.0, (ones(5), ones(5)))

    @test zerod_container([[1.0], [1.0]]) == [[0.0], [0.0]]
    @test oned_container([[5.0], [4.0]]) == [[1.0], [1.0]]

    @test zerod_container(Dict("a"=>5.0, "b"=>randn(3))) == Dict("a"=>0.0, "b"=>zeros(3))
    @test oned_container(Dict("a"=>5.0, "b"=>randn(3))) == Dict("a"=>1.0, "b"=>ones(3))

    @test ref_equal(zerod_container(Ref(4)), Ref(0))
    @test ref_equal(oned_container(Ref(4)), Ref(1))
end

# To ensure we end up using the fallback machinery for ∇(x̄, f, ...) we'll define a new
# function and setup for it to use in the testset below
quad(A::Matrix, B::Matrix) = B'A*B
@explicit_intercepts quad Tuple{Matrix, Matrix}
Nabla.∇(::typeof(quad), ::Type{Arg{1}}, p, Y, Ȳ, A::Matrix, B::Matrix) = B*Ȳ*B'
Nabla.∇(::typeof(quad), ::Type{Arg{2}}, p, Y, Ȳ, A::Matrix, B::Matrix) = A*B*Ȳ' + A'B*Ȳ

@testset "Mutating values in the tape" begin
    rng = MersenneTwister(123456)
    n = 5
    A = Leaf(Tape(), randn(rng, n, n))
    B = randn(rng, n, n)
    Q = quad(A, B)
    QQ = quad(Q, B)
    rt = ∇(QQ, Matrix(1.0I, n, n))
    oldvals = map(deepcopy∘unbox, getfield(rt, :tape))
    Nabla.propagate(Q, rt)  # This triggers a mutating addition
    newvals = map(unbox, getfield(rt, :tape))
    @test !(oldvals[1] ≈ newvals[1])
    @test oldvals[2:end] ≈ newvals[2:end]
end

end # testset "core"
