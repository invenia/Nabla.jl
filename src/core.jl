using BenchmarkTools

import Base: push!, length, show, getindex, setindex!, endof, eachindex
export Tape, Node, Branch, Root, ∇, getzero, getone


# Define functionality to return a type-appropriate zero / one / random element.
@inline dictit(val::Dict, f::Function) = Dict(n => f(val[n]) for n in eachindex(val))
returns_basic = [
    (:AbstractFloat, :(0.0), :(1.0), :(rand() * (ub - lb) + lb)),
    (:AbstractArray, :(zeros(val)), :(ones(val)), :(rand(size(val)) * (ub - lb) + lb)),
    (:(Union{Set, Tuple}), :(map(getzero, val)), :(map(getone, val)), :(map(getrand, val))),
    (:Dict, :(dictit(val, getzero)), :(dictit(val, getone)), :(dictit(val, getrand))),
]
for (dtype, zeroexpr, oneexpr, randexpr) in returns_basic
    @eval @inline getzero(val::$dtype) = $zeroexpr
    @eval @inline getone(val::$dtype) = $oneexpr
    @eval @inline getrand(val::$dtype) = $randexpr
end


abstract Node{T}

immutable Tape
    tape::Vector{Any}
    function Tape()
        tape = new(Vector{Any}())
        sizehint!(tape.tape, 100)
        return tape
    end
    Tape(N::Int) = new(Vector{Any}(N))
end
function show(io::IO, tape::Tape)
    if length(tape) == 0
        println("Empty tape.")
    else
        for n in eachindex(tape)
            println(io, isdefined(tape.tape, n) ? tape[n] : "#undef")
        end
    end
end
@inline getindex(tape::Tape, n::Int) = tape.tape[n]
@inline getindex(tape::Tape, node::Node) = getindex(tape, node.pos)
@inline endof(tape::Tape) = length(tape)
@inline setindex!(tape::Tape, x, n::Int) = (tape.tape[n] = x; tape)
@inline eachindex(tape::Tape) = eachindex(tape.tape)
@inline length(tape::Tape) = length(tape.tape)
@inline push!(tape::Tape, node::Node) = (push!(tape.tape, node); tape)


"""
An element at the 'bottom' of the computational graph.

Fields:
val - the value of the node.
tape - The Tape to which this Root is assigned.
pos - the location of this Root in the tape to which it is assigned.
"""
immutable Root{T} <: Node{T}
    val::T
    tape::Tape
    pos::Int
end
function Root(val, tape::Tape)
    root = Root(val, tape, length(tape) + 1)
    push!(tape, root)
    return root
end
function show{T}(io::IO, tape::Root{T})
    print(io, "Root{$T} $(tape.val)")
end


"""
A Branch is a Node with parents (args).

Fields:
val - the value of this node produced in the forward pass.
f - the function used to generate this Node.
args - Values indicating which elements in the tape will require updating by this node.
tape - The Tape to which this Branch is assigned.
pos - the location of this Branch in the tape to which it is assigned.
"""
immutable Branch{T, V <: Tuple} <: Node{T}
    val::T
    f::Function
    args::V
    tape::Tape
    pos::Int
end
@inline function Branch(f::Function, args::Tuple, tape::Tape)
    branch = Branch(f(map(unbox, args)...), f, args, tape, length(tape) + 1)
    push!(tape, branch)
    return branch
end
function show{T, V}(io::IO, branch::Branch{T, V})
    print(io, "Branch{$T} $(branch.val), f=$(branch.f)")
end


# Location of Node on tape. -1 if not a Node object.
pos(x::Node) = x.pos
pos(x) = -1

# Get the value from the Node if the passed object is a Node.
unbox(x::Node) = x.val
unbox(x) = x

# Roots do nothing, Branches compute their own sensitivities and update others.
@inline propagate_sensitivities(y::Root, δ::Int, rvs_tape::Tape) = nothing
@inline function propagate_sensitivities(y::Branch, δ::Int, rvs_tape::Tape)
    a = map(unbox, y.args)
    b = map(pos, y.args)
    y.f(rvs_tape, y.val, rvs_tape.tape[y.pos], a..., b...)::Void
    return nothing
end

"""
Perform the reverse pass.

Inputs:
node - The Node w.r.t. which we will computed gradients.

Outputs:
a Tape containing the reverse-mode sensitivities w.r.t. node of every node in node.tape.
"""
function ∇(node::Node)

    # Construct reverse tape and initialise the last element.
    fwd_tape, rvs_tape = node.tape, Tape(node.pos)
    rvs_tape[end] = getone(node.val)

    # Iterate backwards through the reverse tape and return the result.
    for n in eachindex(rvs_tape)
        δ = node.pos - n + 1
        isdefined(rvs_tape.tape, δ) && propagate_sensitivities(fwd_tape[δ], δ, rvs_tape)
    end
    return rvs_tape
end
