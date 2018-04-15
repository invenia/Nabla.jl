import Base: push!, length, show, getindex, setindex!, lastindex, eachindex, isassigned
export Leaf, Tape, Node, Branch, ∇, ∇Ctx, propagate_forward

""" Basic unit on the computational graph."""
abstract type Node{T} end

""" A topologically ordered collection of Nodes. """
struct Tape
    tape::Vector{Any}
    Tape() = new(Vector{Any}())
    Tape(N::Int) = new(Vector{Any}(undef, N))
end
function show(io::IO, tape::Tape)
    if length(tape) == 0
        println(io, "Empty tape.")
    else
        for n in eachindex(tape)
            println(io, n, " ", isassigned(tape.tape, n) ? tape[n] : "#undef")
        end
    end
end
@inline getindex(tape::Tape, n::Int) = Base.getindex(tape.tape, n)
@inline getindex(tape::Tape, node::Node) = Base.getindex(tape, node.pos)
@inline lastindex(tape::Tape) = length(tape)
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
function Leaf(tape::Tape, val)
    leaf = Leaf(val, tape, length(tape) + 1)
    push!(tape, leaf)
    return leaf
end
show(io::IO, tape::Leaf{T}) where T = print(io, "Leaf{$T} $(tape.val)")
show(io::IO, tape::Leaf{T}) where T<:AbstractArray = print(io, "Leaf{$T} $(size(tape.val))")

"""
A Branch is a Node with parents (args).

Fields:
val - the value of this node produced in the forward pass.
f - the function used to generate this Node.
args - Values indicating which elements in the tape will require updating by this node.
tape - The Tape to which this Branch is assigned.
pos - the location of this Branch in the tape to which it is assigned.
"""
struct Branch{T} <: Node{T}
    val::T
    f
    args::Tuple
    tape::Tape
    pos::Int
end
function Branch(f, args::Tuple, tape::Tape)
    unboxed = unbox.(args)
    branch = Branch(f(unboxed...), f, args, tape, length(tape) + 1)
    push!(tape, branch)
    return branch
end
show(io::IO, branch::Branch{T}) where T =
    print(io, "Branch{$T} $(branch.val) f=$(branch.f)")
show(io::IO, branch::Branch{T}) where T<:AbstractArray =
    print(io, "Branch{$T} $(size(branch.val)) f=$(branch.f)")

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

"""
    propagate_forward(f, args...)

Continue execution as usual if no `Node`s are encountered. Otherwise track execution.
"""
@generated function propagate_forward(f, args...)
    tape_arg = findfirst(arg->arg<:Node, args)
    return tape_arg === nothing ? :(f(args...)) : :(Branch(f, args, args[$tape_arg].tape))
end

# Leafs do nothing, Branches compute their own sensitivities and update others.
@inline propagate(y::Leaf, rvs_tape::Tape) = nothing
function propagate(y::Branch, rvs_tape::Tape)
    tape = rvs_tape.tape
    ȳ, f = tape[y.pos], y.f
    xs, xids = map(unbox, y.args), map(pos, y.args)
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

function propagate(fwd_tape::Tape, rvs_tape::Tape)
    for n in eachindex(rvs_tape)
        δ = length(rvs_tape) - n + 1
        isassigned(rvs_tape.tape, δ) && propagate(fwd_tape[δ], rvs_tape)
    end
    return rvs_tape
end


""" Initialise a Tape appropriately for being used as a reverse-tape. """
function reverse_tape(y::Node, ȳ)
    tape = Tape(y.pos)
    tape[end] = ȳ
    return tape
end

""" Used to flag which argument is being specified in x̄. """
const Arg{N} = Val{N}

"""
    ∇(y::Node{<:∇Scalar})
    ∇(y::Node{T}, ȳ::T) where T

Return a `Tape` object which can be indexed using `Node`s, each element of which contains
the result of multiplying `ȳ` by the transpose of the Jacobian of the function specified by
the `Tape` object in `y`. If `y` is a scalar and `ȳ = 1` then this is equivalent to
computing the gradient of `y` w.r.t. each of the elements in the `Tape`.


    ∇(f::Function, ::Type{Arg{N}}, p, y, ȳ, x...)
    ∇(x̄, f::Function, ::Type{Arg{N}}, p, y, ȳ, x...)

To implement a new reverse-mode sensitivity for the `N^{th}` argument of function `f`. p
is the output of `preprocess`. `x1`, `x2`,... are the inputs to the function, `y` is its
output and `ȳ` the reverse-mode sensitivity of `y`.
"""
∇(y::Node, ȳ) = propagate(y.tape, reverse_tape(y, ȳ))
@inline ∇(y::Node{<:∇Scalar}) = ∇(y, one(y.val))

@inline ∇(x̄, f, ::Type{Arg{N}}, args...) where N = x̄ + ∇(f, Arg{N}, args...)

"""
    ∇(f; get_output::Bool=false)

Returns a function which, when evaluated with arguments that are accepted by `f`, will
return the gradient w.r.t. each of the arguments.
"""
function ∇(f; get_output::Bool=false)
    return function(args...)
        args_ = Leaf.(Ref(Tape()), args)
        y = overdub(∇Ctx, f)(args_...)
        y isa Node || throw(error("f is not a function of its arguments."))
        typeof(y.val) <: ∇Scalar || throw(error("output is not scalar."))
        ∇f = ∇(y)
        ∇args = ([∇f[arg_] for arg_ in args_]...,)
        return get_output ? (y, ∇args) : ∇args
    end
end

__ones(x) = fill(one(eltype(x)), size(x))

# A collection of methods for initialising nested indexable containers to zero.
for (f_name, scalar_init, array_init) in
    zip((:zerod_container, :oned_container, :randned_container),
        (:zero, :one, Nullable()),
        (:zero, :__ones, Nullable()))
    if !isnull(scalar_init)
        @eval @inline $f_name(x::Number) = $scalar_init(x)
    end
    if !isnull(array_init)
        @eval @inline $f_name(x::AbstractArray{<:Real}) = $array_init(x)
    end
    eval(quote
        @inline $f_name(x::Tuple) = ([$f_name(n) for n in x]...,)
        @inline function $f_name(x)
            y = Base.copy(x)
            for n in eachindex(y)
                @inbounds y[n] = $f_name(y[n])
            end
            return y
        end
    end)
end
@inline randned_container(x::Number) = randn(typeof(x))
@inline randned_container(x::AbstractArray{<:Real}) = randn(eltype(x), size(x)...)
for T in (:Diagonal, :UpperTriangular, :LowerTriangular)
    @eval @inline randned_container(x::$T{<:Real}) = $T(randn(eltype(x), size(x)...))
end

"""
    preprocess(::Function, args...)

Default implementation of preprocess returns an empty Tuple. Individual sensitivity
implementations should add methods specific to their use case. The output is passed
in to `∇` as the 3rd or 4th argument in the new-x̄ and update-x̄ cases respectively.
"""
@inline preprocess(::Any, args...) = ()
