import Base: push!, length, show, getindex, setindex!, lastindex, eachindex, isassigned
export ∇, ∇Ctx, Forward

using Cassette: Box, unbox, meta, overdub, mapcall, Context
@context ∇Ctx

abstract type Node end
const Tape = Vector{Node}

struct PlainData{Ty} <: Node
    val::Ty
    n::Int
end
struct Leaf{Ty} <: Node
    val::Ty
    n::Int
end
struct Branch{Ty, N} <: Node
    val::Ty
    args::Tuple{Vararg{Int, N}}
    n::Int
end

pos!(::Tape, x::Node) = x.n
function pos!(tape::Tape, x)
    push!(tape, PlainData(x, length(tape) + 1))
    return length(tape)
end

@generated propagate(ctx::∇Ctx, tape::Tape, f, args...) =
    any(map(x->x<:Box, args)) ? :(_propagate(ctx, tape, f, args...)) : :(f(args...))

# Compute the and track of the operation implied by `f` and `args`. Track anything that
# isn't already tracked, including plain (non-boxed) data, for use on the reverse-pass.
function _propagate(ctx::∇Ctx, tape::Tape, f, args...)
    push!(tape, Branch(f(args...), map(x->pos!(tape, x), (f, args...)), length(tape) + 1))
    return Box(ctx, tape[end], length(tape))
end

"""
    forward(f, x...)

Perform a forward pass through `f`, keeping track of the information required for the
reverse-pass. Returns a `Tape` whos last element contains the output of the function.
"""
function forward(f, x...)
    c, tape = ∇Ctx(f), push!(Tape(), map(n->Leaf(x[n], n), eachindex(x))...)
    overdub(c, f; metadata=tape)(Box.(Ref(c), tape, eachindex(x))...)
    return tape
end

∇(::typeof(forward), ::Type{Val{N}}, rvs::Vector{Any}, y::Tape, ȳ, f, x...) where N = rvs[N]
function preprocess(::typeof(forward), tape::Tape, ȳ, f, x::Vararg{Any, N}) where N
    @assert all(x .=== tape[1:N])
    rvs = setindex!(Vector{Any}(length(tape)), ȳ, length(tape))
    for n in reverse(eachindex(tape))
        if isassigned(rvs[n]) && typeof(tape[n]) <: Branch
            fidx, xidxs = tape[n].args[1], tape[n].args[2:end]
            f, y, ȳ = tape[fidx].val, tape[n].val, rvs[n]
            xs = map(x->x.val, tape[fidxs])
            p = preprocess(f, y, ȳ, xs...)
            for (n′, (x, xid)) in enumerate(zip(xs, xidxs))
                if !(typeof(x) <: PlainData)
                    rvs[xid] = isassigned(rvs, xid) ?
                        ∇(rvs[xid], f, Val{n′}, p, y, ȳ, xs...) :
                        ∇(f, Val{n′}, p, y, ȳ, xs...)
                end
            end
        end
    end
    return rvs
end

"""
    ∇(f)

Return a function which accepts the same arguments as `f`, but returns the gradient of `f`.

    ∇(f::Function, ::Type{Val{N}}, p, y, ȳ, x...)
    ∇(x̄, f::Function, ::Type{Val{N}}, p, y, ȳ, x...)

To implement a new reverse-mode sensitivity for the `N^{th}` argument of function `f`. p
is the output of `preprocess`. `x1`, `x2`,... are the inputs to the function, `y` is its
output and `ȳ` the reverse-mode sensitivity of `y`.
"""
function ∇(f)
    return function(x...)
        tape = forward(f, x...)
        @assert typeof(tape[end].val) <: Number
        ȳ = one(tape[end].val)
        rvs = preprocess(forward, tape, ȳ, f, x...)
        return map(∇(forward, Val{n}, rvs, tape, ȳ, f, x...), eachindex(x))
    end
end


@inline ∇(x̄, f, ::Type{Val{N}}, args...) where N = x̄ + ∇(f, Val{N}, args...)

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
@inline preprocess(::Any, args...) = nothing
