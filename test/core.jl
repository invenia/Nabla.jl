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
end # testset "core"
