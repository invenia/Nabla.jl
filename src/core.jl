import Base: push!, length, show, getindex, setindex!, eachindex, isassigned,
             isapprox, zero, one, lastindex

export Leaf, Tape, Node, Branch, ∇

""" Basic unit on the computational graph."""
abstract type Node{T} end

""" A topologically ordered collection of Nodes. """
struct Tape
    tape::Vector{Any}
    Tape() = new(Vector{Any}())
    Tape(N::Int) = new(Vector{Any}(undef, N))
end
function show(io::IO, t::Tape)
    n = length(t)
    print(io, "Tape with ", n, " element", n == 1 ? "" : "s", n > 0 ? ":" : "")
    for i in eachindex(t)
        print(io, "\n  [", i, "]: ")
        if isassigned(tape(t), i)
            show(io, t[i])
        else
            print(io, "#undef")
        end
    end
end
@inline getindex(t::Tape, n::Int) = unthunk(getindex(tape(t), n))
@inline getindex(t::Tape, node::Node) = getindex(t, pos(node))
@inline lastindex(t::Tape) = length(t)
@inline setindex!(t::Tape, x, n::Int) = (tape(t)[n] = x; t)
@inline eachindex(t::Tape) = eachindex(tape(t))
@inline length(t::Tape) = length(tape(t))
@inline push!(t::Tape, node::Node) = (push!(tape(t), node); t)
@inline isassigned(t::Tape, n::Int) = isassigned(tape(t), n)
@inline isassigned(t::Tape, node::Node) = isassigned(t, pos(node))

# Make `Tape`s broadcast as scalars without a warning on 0.7
Base.Broadcast.broadcastable(tape::Tape) = Ref(tape)

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
show(io::IO, tape::Leaf{T}) where T = print(io, "Leaf{$T} $(unbox(tape))")
show(io::IO, tape::Leaf{T}) where T<:AbstractArray = print(io, "Leaf{$T} $(size(unbox(tape)))")

"""
A `Branch` is a Node with parents (args).

Fields:
val::T - the value of this node produced in the forward pass.
f - the function used to generate this Node.
args - Values indicating which elements in the tape will require updating by this node.
tape - The Tape to which this Branch is assigned.
pos - the location of this Branch in the tape to which it is assigned.
pullback::B - if there is a custom primate rule (a `ChainRulesCore.rrule`) then this holds
    the pullback to propagates gradients back through the operation, if there is not a rule
    then this is set to `nothing`.
    It also maybe set to `nothing` by legacy Nabla rules that have not moved to ChainRules.
"""
struct Branch{T, B} <: Node{T}
    val::T
    f
    args::Tuple
    kwargs::NamedTuple
    tape::Tape
    pos::Int
    pullback::B
end
function Branch(f, args::Tuple, tape::Tape; kwargs...)
    unboxed = unbox.(args)

    # We could check for an `rrule` here if we wanted but we don't,
    # because we should never reach this point if we have an rrule
    primal_val = f(unboxed...; kwargs...)
    pullback = nothing

    branch = Branch(primal_val, f, args, kwargs.data, tape, length(tape) + 1, pullback)
    push!(tape, branch)
    return branch
end
show(io::IO, branch::Branch{T}) where T =
    print(io, "Branch{$T} $(unbox(branch)) f=$(getfield(branch, :f))")
show(io::IO, branch::Branch{T}) where T<:AbstractArray =
    print(io, "Branch{$T} $(size(unbox(branch))) f=$(getfield(branch, :f))")

"""
    tape(x::Node)
    tape(x::Tape)

Retrieve the `Tape` in a `Node`, or the underyling vector in a `Tape`.
"""
tape(x::Union{Node,Tape}) = getfield(x, :tape)

"""
    pos(x::Node)
    pos(x)

Location of Node on tape. -1 if not a Node object.
"""
pos(x::Node) = getfield(x, :pos)
pos(x) = -1

"""
    unbox(x::Node)
    unbox(x)

Get `.val` if `x` is a Node, otherwise is equivalent to `identity`.
"""
unbox(x::Node) = getfield(x, :val)
unbox(x) = x

isapprox(n::Node, f) = unbox(n) ≈ f
isapprox(f, n::Node) = n ≈ f
isapprox(n::Node, f::Node) = unbox(n) ≈ unbox(f)

zero(n::Node) = zero(unbox(n))
one(n::Node) = one(unbox(n))

# Let the user get the `size` and `length` of `Node`s.
Base.size(x::Node, dims...) = size(unbox(x), dims...)
Base.length(x::Node) = length(unbox(x))

# Leafs do nothing, Branches compute their own sensitivities and update others.
@inline propagate(y::Leaf, rvs_tape::Tape) = nothing
function propagate(y::Branch, rvs_tape::Tape)
    ȳ = rvs_tape[y]  # the gradient we are going to propagate through the operation in y
    d_tape = Nabla.tape(rvs_tape)  # strips off the Tape abstration leaving a plain Vector
    f = getfield(y, :f)
    args = getfield(y, :args)
    kwargs = getfield(y, :kwargs)
    xs = map(unbox, args)
    xids = map(pos, args)
    p = preprocess(f, y, ȳ, args...)  # inlining CSE will avoid unboxing twice.
    for j in eachindex(xs)
        x, xid = xs[j], xids[j]
        if xid > 0
            d_tape[xid] = isassigned(d_tape, xid) ?
                ∇(d_tape[xid], f, Arg{j}, p, unbox(y), ȳ, xs...; kwargs...) :  # maybe-inplace version
                ∇(f, Arg{j}, p, unbox(y), ȳ, xs...; kwargs...)
        end
    end
    return nothing
end

function propagate(fwd_tape::Tape, rvs_tape::Tape)
    for n in eachindex(rvs_tape)
        δ = length(rvs_tape) - n + 1
        isassigned(tape(rvs_tape), δ) && propagate(fwd_tape[δ], rvs_tape)
    end
    return rvs_tape
end


""" Initialise a Tape appropriately for being used as a reverse-tape. """
function reverse_tape(y::Node, ȳ)
    tape = Tape(pos(y))
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

To implement a new reverse-mode sensitivity for the `N^{th}` argument of function `f`. p
is the output of `preprocess`. `x1`, `x2`,... are the inputs to the function, `y` is its
output and `ȳ` the reverse-mode sensitivity of `y`.

∇(x̄, f::Function, ::Type{Arg{N}}, p, y, ȳ, x...)
This is the optionally inplace version of `∇` that should, if implemented, mutate
x̄ to have the gradient added to it.
"""
∇(y::Node, ȳ) = propagate(tape(y), reverse_tape(y, ȳ))
@inline ∇(y::Node{<:∇Scalar}) = ∇(y, one(unbox(y)))

@inline function ∇(x̄, f, ::Type{Arg{N}}, args...; kwargs...) where N
    return ChainRulesCore.add!!(x̄, ∇(f, Arg{N}, args...; kwargs...))
end

"""
    ∇(f; get_output::Bool=false)

Returns a function which, when evaluated with arguments that are accepted by `f`, will
return the gradient w.r.t. each of the arguments. If `get_output` is `true`, the result
of calling `f` on the given arguments is also returned.
"""
function ∇(f; get_output::Bool=false)
    return function(args...; kwargs...)
        args_ = Leaf.(Tape(), args)
        y = f(args_...; kwargs...)
        if y isa Node
            ∇f = ∇(y)
            ∇args = map(args_, args) do arg_, arg
                isassigned(∇f, arg_) ? ∇f[arg_] : zero(arg)
            end
        else
            ∇args = zero.(args)
        end
        ∇args_public = map(unthunk, ∇args)
        return get_output ? (y, ∇args_public) : ∇args_public
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
        (:zero, :one, nothing),
        (:zeros, :ones, nothing))
    if scalar_init !== nothing
        @eval @inline $f_name(x::Number) = $scalar_init(x)
    end
    if array_init !== nothing
        @eval @inline $f_name(x::AbstractArray{<:Real}) = $array_init(eltype(x), size(x))
    end
    eval(quote
        @inline $f_name(x::Tuple) = map($f_name, x)
        @inline function $f_name(x)
            y = Base.copy(x)
            for n in eachindex(y)
                @inbounds y[n] = $f_name(y[n])
            end
            return y
        end
        $f_name(x::Ref) = Ref($f_name(x[]))
    end)
end
@inline randned_container(x::Number) = randn(typeof(x))
@inline randned_container(x::AbstractArray{<:Real}) = randn(eltype(x), size(x)...)
for T in (:Diagonal, :UpperTriangular, :LowerTriangular)
    @eval @inline randned_container(x::$T{<:Real}) = $T(randn(eltype(x), size(x)...))
end

# Bare-bones FMAD implementation based on internals of ForwardDiff.
# Accepts a Tuple of args and returns a Tuple of gradients.
# Currently scales almost exactly linearly with the number of inputs.
# The coefficient of this scaling could be improved by fully utilizing ForwardDiff
# and computing from multiple seeds at the same time.
function dual_call_expr(f, x::Type{<:Tuple}, ::Type{Type{Val{n}}}) where n
    dual_call = Expr(:call, :f)
    for m in 1:Base.length(x.parameters)
        push!(dual_call.args, n == m ? :(ForwardDiff.Dual(x[$m], 1)) : :(x[$m]))
    end
    return :(first(ForwardDiff.partials($dual_call)))
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
