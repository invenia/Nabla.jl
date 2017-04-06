import Base: push!, length, show, getindex, setindex!, endof, eachindex
export Tape, Node, Branch, Root, ∇

abstract Node{T}

type Tape
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
function Branch(f::Function, args::Tuple, tape::Tape)
    branch = Branch(f(map(unbox, args)...), f, args, tape, length(tape) + 1)
    push!(tape, branch)
    return branch
end
function show{T, V}(io::IO, branch::Branch{T, V})
    print(io, "Branch{$T} $(branch.val), f=$(branch.f)")
end
push!(tape::Tape, node::Node) = (push!(tape.tape, node); tape)
length(tape::Tape) = length(tape.tape)

# Get the value from the Node if the passed object is a Node.
@inline unbox(x::Node) = x.val
@inline unbox(x) = x

# Update the gradient accumulator if it's a Node.
function accumulate!{T}(x::Node{T}, darg::T, reverse_tape::Tape)
    n = x.pos
    reverse_tape[n] = isdefined(reverse_tape.tape, n) ? reverse_tape[n] + darg : darg
    return nothing
end
accumulate!{T, V}(x::Node{T}, darg::V, rvs) = error("Type of val and dval not the same.")
accumulate!(x, darg, rvs) = nothing


"""
Perform the reverse pass.

Inputs:
tape - a tape object. Forward pass will already have been done.

Outputs:
reverse-mode sensitivities w.r.t. every Node.
"""
@inline ∇(node::Node) = ∇(node.tape, node.pos)
@inline ∇(tape::Tape) = ∇(tape, length(tape))
function ∇(forward_tape::Tape, N::Int)

    N > length(forward_tape) && throw(ArgumentError("N > length(tape)."))

    # Construct reverse tape and initialise the last element.
    reverse_tape = Tape(length(forward_tape))
    reverse_tape[end] = getone(forward_tape.tape[end].val)

    # Roots do nothing, Branches compute their own sensitivities and update others.
    g(y::Root, n::Int, N::Int) = nothing
    function g(y::Branch, n::Int, N::Int)
        !isdefined(reverse_tape.tape, N - n) && return
        dargs = y.f(y, y.val, reverse_tape[N - n], map(unbox, y.args)...)
        for (arg, darg) in zip(y.args, dargs)
            accumulate!(arg, darg, reverse_tape)
        end
    end

    # Iterate backwards through the reverse tape.
    N = length(forward_tape) + 1
    for n in eachindex(reverse_tape)
        g(forward_tape.tape[N - n], n, N)
    end

    # Extract 
    return reverse_tape
end
