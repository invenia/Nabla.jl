using DualNumbers

import Base: push!, length, show, getindex, setindex!, endof, eachindex, isassigned
export Leaf, Tape, Node, Branch, ∇

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
            println(io, n, " ", isassigned(tape.tape, n) ? tape[n] : "#undef")
        end
    end
end
@inline getindex(tape::Tape, n::Int) = Base.getindex(tape.tape, n)
@inline getindex(tape::Tape, node::Node) = Base.getindex(tape, node.pos)
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
immutable Branch{T} <: Node{T}
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
struct Arg{N} end

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
function ∇(f, get_output::Bool=false)
    return function(args...)
        args_ = Leaf.(Tape(), args)
        y = f(args_...)
        y isa Node || return zero.(args)
        ∇f = ∇(y)
        ∇args = ([isassigned(∇f, arg_) ? ∇f[arg_] : zero(arg)
		for (arg_, arg) in zip(args_, args)]...)
	return get_output ? (y, ∇args) : ∇args
    end
end

# """
#     ∇(f::Function)

# Returns a function which, when evaluated with arguments that are accepted by `f` (`x`),
# will return a Tuple, the first element of which is the output of the function `f` and then
# second element of which is (yet another) function `g`. `g` can either be evaluated with no
# arguments, in which case it will return the gradient of `f` evaluated at `x`.
# Alternatively, it can be evaluated with arguments of the same type and shape as the output
# of `f(x)`, in which case it is equivalent to multiplying them 'from the left' by the
# Jacobian ∂(f(x)) / ∂x.
# """
# function ∇(f::Function)
#     return function(args...)
#         args_ = Leaf.(Tape(), args)
#         y = f(args_...)
#         ∇fx = (ȳ)->∇

#     end
# end

# A collection of methods for initialising nested indexable containers to zero.
for (f_name, scalar_init, array_init) in
    zip((:zerod_container, :oned_container, :randned_container),
        (:zero, :one, Nullable()),
        (:zeros, :ones, Nullable()))
    if !isnull(scalar_init)
        @eval @inline $f_name(x::Number) = $scalar_init(x)
    end
    if !isnull(array_init)
        @eval @inline $f_name(x::AbstractArray{<:Real}) = $array_init(x)
    end
    eval(quote
        @inline $f_name(x::Tuple) = ([$f_name(n) for n in x]...)
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

# Bare-bones FMAD implementation based on DualNumbers. Accepts a Tuple of args and returns
# a Tuple of gradients. Currently scales almost exactly linearly with the number of inputs.
# The coefficient of this scaling could be improved by implementing a version of DualNumbers
# which computes from multiple seeds at the same time.
function dual_call_expr(f, x::Type{<:Tuple}, ::Type{Type{Val{n}}}) where n
    dual_call = Expr(:call, :f)
    for m in 1:Base.length(x.parameters)
        push!(dual_call.args, n == m ? :(Dual(x[$m], 1)) : :(x[$m]))
    end
    return :(dualpart($dual_call))
end
@generated fmad(f, x, n) = dual_call_expr(f, x, n)
function fmad_expr(f, x::Type{<:Tuple})
    body = Expr(:tuple)
    for n in 1:Base.length(x.parameters)
        push!(body.args, dual_call_expr(f, x, Type{Val{n}}))
    end
    return body
end
@generated fmad(f, x) = fmad_expr(f, x)
