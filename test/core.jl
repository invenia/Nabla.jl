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
    leaf = Leaf(5, Tape())
    @test isassigned(leaf.tape.tape, leaf.pos) == isassigned(leaf.tape, leaf)

    # Simple tests for `Leaf`.
    tp1, tp2 = Tape(), Tape(50)
    @test Leaf(5.0, tp1).tape == tp1
    @test Leaf(5.0, tp1).val == 5.0
    @test Leaf(5.0, tp1).pos == 3
    @test Leaf(5, tp2).pos == 51
    @test Leaf(5, tp2).val == 5
    @test Leaf(5, tp2).tape == tp2

    # Simple tests for `Branch`.
    foo_coeff = 10
    foo(x::Real) = foo_coeff * x
    foo(x::Node{T} where T<:Real) = Branch(foo, (x,), x.tape)
    DiffCore.∇(::typeof(foo), ::Type{Arg{1}}, p, y, ȳ, x) = foo_coeff
    function get_new_branch()
        leaf = Leaf(5, Tape())
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
    @test DiffCore.pos(1) == -1
    @test DiffCore.pos("hi") == -1
    @test DiffCore.pos(Leaf(5, Tape())) == 1
    @test DiffCore.pos(Leaf(5.0, Tape(50))) == 51

    # Simple tests for `unbox`.
    @test DiffCore.unbox(1) == 1
    @test DiffCore.unbox("hi") == "hi"
    @test DiffCore.unbox(Leaf(5, Tape())) == 5
    @test DiffCore.unbox(Leaf(5.0, Tape())) == 5.0

    # Simple tests for reverse_tape.
    @test isa(DiffCore.reverse_tape(Leaf(5, Tape())), Tape)
    @test length(DiffCore.reverse_tape(Leaf(5, Tape()))) == 1
    @test DiffCore.reverse_tape(Leaf(5, Tape()))[end] == 1
    @test isassigned(DiffCore.reverse_tape(Leaf(5, Tape())), 1)
    @test isassigned(DiffCore.reverse_tape(get_new_branch()), 2)
    @test !isassigned(DiffCore.reverse_tape(get_new_branch()), 1)

    # Simple tests for `propagate`.
    @test DiffCore.propagate(Leaf(5, Tape()), Tape()) == nothing
    br = get_new_branch()
    rvs_tape = DiffCore.reverse_tape(br)
    @test DiffCore.propagate(br, rvs_tape) == nothing
    @test rvs_tape[br] == 1
    @test rvs_tape[br.args[1]] == foo_coeff

    # Simple integration tests for ∇.
    br = get_new_branch()
    @test ∇(br)[br] == 1
    @test ∇(br)[br.args[1]] == foo_coeff

end # let
end # testset "core"
