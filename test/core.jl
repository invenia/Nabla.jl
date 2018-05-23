@testset "core" begin

let
    # Simple tests for `Tape`.
    @test getindex(setindex!(Tape(5), "hi", 5), 5) == "hi"
    @test endof(Tape(50)) == 50
    @test eachindex(Tape(50)) == Base.OneTo(50)
    @test length(Tape()) == 0
    @test length(Tape(50)) == 50
    @test !isassigned(Tape(1), 1)
    @test !isassigned(Tape(26), 26)

    # Check that printing works as expected.
    let
        buffer = IOBuffer()
        show(buffer, Tape())
        @test String(buffer) == "Empty tape.\n"
    end
    let
        buffer = IOBuffer()
        show(buffer, Tape(1))
        @test String(buffer) == "1 #undef\n"
    end
    let
        buffer = IOBuffer()
        show(buffer, Tape(2))
        @test String(buffer) == "1 #undef\n2 #undef\n"
    end
    let
        buffer = IOBuffer()
        tape = Tape(1)
        tape[1] = 5
        show(buffer, tape)
        @test String(buffer) == "1 5\n"
    end

    # Check isassigned consistency.
    leaf = Leaf(Tape(), 5)
    @test isassigned(leaf.tape.tape, leaf.pos) == isassigned(leaf.tape, leaf)

    # Simple tests for `Leaf`.
    tp1, tp2 = Tape(), Tape(50)
    @test Leaf(tp1, 5.0).tape == tp1
    @test Leaf(tp1, 5.0).val == 5.0
    @test Leaf(tp1, 5.0).pos == 3
    @test Leaf(tp2, 5).pos == 51
    @test Leaf(tp2, 5).val == 5
    @test Leaf(tp2, 5).tape == tp2

    # Simple tests for `Branch`.
    foo_coeff = 10
    foo(x::Real) = foo_coeff * x
    foo(x::Node{T} where T<:Real) = Branch(foo, (x,), x.tape)
    Nabla.∇(::typeof(foo), ::Type{Arg{1}}, p, y, ȳ, x) = ȳ * foo_coeff
    function get_new_branch()
        leaf = Leaf(Tape(), 5)
        return foo(leaf)
    end
    @test isa(get_new_branch(), Branch)
    @test get_new_branch().val == 50
    @test get_new_branch().f == foo
    @test length(get_new_branch().args) == 1
    @test isa(get_new_branch().args[1], Leaf)
    @test get_new_branch().args[1].val == 5
    @test get_new_branch().args[1].pos == 1
    @test get_new_branch().pos == 2
    @test get_new_branch().args[1].tape != get_new_branch().tape
    branch = get_new_branch()
    @test branch.tape == branch.args[1].tape

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

end # let

# Check that functions involving `isapprox` can be differentiated
let
    f(x) = x ≈ 5.0 ? 1.0 : 3.0 * x
    g(x) = 5.0 * x
    h(x) = g(x) ≈ 25.0 ? x : f(x) + g(x)
    ∇f = ∇(f)
    ∇h = ∇(h)
    @test ∇f(5.0) == (0.0,)
    @test ∇f(6.0) == (3.0,)
    @test ∇h(5.0) == (1.0,)
    @test ∇h(6.0) == (8.0,)
    f(x) = x ≈ [5.0] ? 1.0 : 3.0 * sum(x)
    ∇f = ∇(f)
    @test ∇f([5.0]) == ([0.0],)
    @test ∇f([6.0]) == ([3.0],)
end

# Check that functions with extra, unused variables can be differentiated
let
    f(a,b,c,d) = a*c
    ∇f = ∇(f)
    g(a,b) = 12
    ∇g = ∇(g)

    @test ∇f(1,2,3,4) == (3, 0, 1, 0)
    @test ∇f(1,[2.0],3,4.0) == (3, [0.0], 1, 0.0)
    @test ∇g(1,2) == (0,0)
end

# Check that functions with `zero` and `one` can be differentiated
let
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

# Check that the convenience implementation of ∇ works as intended.
let
    f(x, y) = 2x + y
    ∇f = ∇(f)
    ∇f_out = ∇(f, true)

    @test_throws MethodError ∇f(randn(5), randn(5))
    x, y = randn(), randn()
    ∇z = ∇(f(Leaf.(Tape(), (x, y))...))
    @test ∇f(x, y) == (∇z[1], ∇z[2])
    z, (∇x, ∇y) = ∇f_out(x, y)
    @test z.val == f(x, y)
    @test (∇x, ∇y) == ∇f(x, y)
end

# Tests for zero'd and one'd containers.
let
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
end

end # testset "core"
