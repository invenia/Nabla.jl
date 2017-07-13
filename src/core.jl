using BenchmarkTools

import Base: push!, length, show, getindex, setindex!, endof, eachindex, isassigned
export Leaf, Tape, Node, Branch, ∇

@inline get_ones(x::Real) = one(x)
@inline get_ones(x::AbstractArray) = ones(x)

""" Basic unit on the computational graph."""
abstract type Node{T} end

""" A topologically ordered collection of Nodes. """
immutable Tape
    tape::Vector{Any}
    Tape() = new(Vector{Any}())
    Tape(N::Int) = new(Vector{Any}(N))
end
function show(io::IO, tape::Tape)
    if length(tape) == 0
        println(io, "Empty tape.")
    else
        for n in eachindex(tape)
            println(io, isassigned(tape.tape, n) ? tape[n] : "#undef")
        end
    end
end
@inline getindex(tape::Tape, n::Int) = Base.getindex(tape.tape, n)
@inline getindex(tape::Tape, node::T where T<:Node) = Base.getindex(tape, node.pos)
@inline endof(tape::Tape) = length(tape)
@inline setindex!(tape::Tape, x, n::Int) = (tape.tape[n] = x; tape)
@inline eachindex(tape::Tape) = eachindex(tape.tape)
@inline length(tape::Tape) = length(tape.tape)
@inline push!(tape::Tape, node::Node) = (push!(tape.tape, node); tape)
@inline isassigned(tape::Tape, n::Int) = isassigned(tape.tape, n)
@inline isassigned(tape::Tape, node::Node) = isassigned(tape, node.pos)

"""
An element at the 'bottom' of the computational graph.

Fields:
val - the value of the node.
tape - The Tape to which this Leaf is assigned.
pos - the location of this Leaf in the tape to which it is assigned.
"""
struct Leaf{T} <: Node{T}
    val::T
    tape::Tape
    pos::Int
end
function Leaf(val, tape::Tape)
    leaf = Leaf(val, tape, length(tape) + 1)
    push!(tape, leaf)
    return leaf
end
show{T}(io::IO, tape::Leaf{T}) = print(io, "Leaf{$T} $(tape.val)")

"""
A Branch is a Node with parents (args).

Fields:
val - the value of this node produced in the forward pass.
f - the function used to generate this Node.
args - Values indicating which elements in the tape will require updating by this node.
tape - The Tape to which this Branch is assigned.
pos - the location of this Branch in the tape to which it is assigned.
"""
immutable Branch{T, F<:Function, V<:Tuple} <: Node{T}
    val::T
    f::F
    args::V
    tape::Tape
    pos::Int
end
@noinline function Branch(f::Function, args::NTuple{N, Any}, tape::Tape) where N
    unboxed = Base.map(arg->get_original(unbox(arg)), args)
    branch = Branch(f(unboxed...), f, args, tape, length(tape) + 1)
    push!(tape, branch)
    return branch
end
function show{T, V}(io::IO, branch::Branch{T, V})
    print(io, "Branch{$T} $(branch.val), f=$(branch.f)")
end

"""
    pos(x::Node)
    pos(x)

Location of Node on tape. -1 if not a Node object.
"""
pos(x::Node) = x.pos
pos(x) = -1

"""
    unbox(x::Node)
    unbox(x)

Get `.val` if `x` is a Node, otherwise is equivalent to `identity`.
"""
unbox(x::Node) = x.val
unbox(x) = x

# Leafs do nothing, Branches compute their own sensitivities and update others.
@inline propagate(y::Leaf, rvs_tape::Tape) = nothing
function propagate(y::Branch{T}, rvs_tape::Tape) where T
    tape = rvs_tape.tape
    ȳ, f = tape[y.pos]::T, y.f
    xs, xids = Base.map(unbox, y.args), Base.map(pos, y.args)
    p = preprocess(f, y.val, ȳ, xs...)
    for j in eachindex(xs)
        x, xid = xs[j], xids[j]
        if xid > 0
            tape[xid] = isassigned(tape, xid) ?
                ∇(tape[xid], f, Arg{j}, p, y.val, ȳ, xs...) :
                ∇(f, Arg{j}, p, y.val, ȳ, xs...)
        end
    end
    return nothing
end

""" Initialise a Tape appropriately for being used as a reverse-tape. """
function reverse_tape(node::Node)
    tape = Tape(node.pos)
    tape[end] = get_ones(node.val)
    return tape
end

""" Used to flag which argument is being specified in x̄. """
struct Arg{N} end

"""
    ∇(::Node)
    ∇(f::Function, ::Type{Arg{N}}, p, y, ȳ, x...)
    ∇(x̄, f::Function, ::Type{Arg{N}}, p, y, ȳ, x...)

To implement a new reverse-mode sensitivity for the `N^{th}` argument of function `f`. p\\
is the output of `preprocess`. `x1`, `x2`,... are the inputs to the function, `y` is its\\
output and `ȳ` the reverse-mode sensitivity of `y`.
"""
function ∇(node::Node)

    # Construct reverse tape and initialise the last element.
    fwd_tape, rvs_tape = node.tape, reverse_tape(node)

    # Iterate backwards through the reverse tape and return the result.
    for n in eachindex(rvs_tape)
        δ = node.pos - n + 1
        isassigned(rvs_tape.tape, δ) && propagate(fwd_tape[δ], rvs_tape)
    end
    return rvs_tape
end
@inline ∇(x̄, f::Function, ::Type{Arg{N}}, args...) where N = x̄ + ∇(f, Arg{N}, args...)
