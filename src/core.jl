import Base: push!, length, show, getindex, setindex!, lastindex, eachindex, isassigned
export ∇, ∇Ctx, forward

# Cassette-stuff. This will presumably need to change as Cassette changes...
using Cassette: Box, unbox, meta, overdub, Context
@context ∇Ctx
Cassette.metatype(::Type{<:∇Ctx}, ::DataType) = Tuple{Int, Tuple}
Cassette.metatype(::Type{<:∇Ctx}, ::Type) = Tuple{Int, Tuple}

# A wrapper to distinguish between integers as data, and integers as tape indices.
struct TapeIdx
    n::Int
end

# A simple tape construction, with a sane printing method.
const Tape = Vector{Box{<:∇Ctx}}
show(io::IO, box::Box{<:∇Ctx}) =
    print(io, "Box{<:∇Ctx}: $(box.meta.data[1]), $(box.value), $(box.meta.data[2])")

"""
    fetch(tape::Tape, x::TapeIdx)
    fetch(::Tape, x::Ref)

Fetch the item `x` from the `tape` if it resides on it (if it's a `TapeIdx`), otherwise
it should be a `Ref`, so dereference it.
"""
fetch(tape::Tape, n::TapeIdx) = tape[n.n].value
fetch(::Tape, x) = x

getref(x::Box{<:∇Ctx}) = TapeIdx(x.meta.data[1])
getref(x) = x

@generated propagate(ctx::∇Ctx, tape::Tape, f, args...) =
    any(map(x->x<:Box, args)) ? :(_propagate(ctx, tape, f, args...)) : :(f(args...))

function _propagate(ctx::∇Ctx, tape::Tape, f, args...)
    val = f(map(arg->unbox(ctx, arg), args)...)
    push!(tape, Box(ctx, val, (length(tape) + 1, map(getref, (f, args...)))))
    return tape[end]
end

"""
    forward(f, x...)

Perform a forward pass through `f`, keeping track of the information required for the
reverse-pass. Returns a `Tape` whos last element contains the output of the function.
"""
function forward(f, x...)
    ctx, tape = ∇Ctx(f), Tape()
    x_boxed = map(n->Box(ctx, x[n], (n, ())), eachindex(x))
    overdub(ctx, f; metadata=push!(tape, x_boxed...))(x_boxed...)
    return tape
end

∇(::typeof(forward), ::Type{Val{N}}, rvs::Vector{Any}, y::Tape, ȳ, f, x...) where N = rvs[N]
function preprocess(::typeof(forward), tape::Tape, ȳ, f, x::Vararg{Any, N}) where N
    @assert all(x .=== map(x->x.value, tape[1:N]))
    rvs = setindex!(Vector{Any}(undef, length(tape)), ȳ, length(tape))
    for n in reverse(eachindex(tape))
        metadata = tape[n].meta.data
        if isassigned(rvs, n) && !isempty(metadata[2])

            # Get data required to compute sensitivities.
            f, xs = metadata[2][1], map(x->fetch(tape, x), metadata[2][2:end])
            y, ȳ = tape[n].value, rvs[n]

            # Do preprocessing and update sensitivities w.r.t. each argument.
            p = preprocess(f, y, ȳ, xs...)
            for (n′, tape_idx) in enumerate(metadata[2][2:end])
                if typeof(tape_idx) <: TapeIdx
                    rvs[tape_idx.n] = isassigned(rvs, tape_idx.n) ?
                        ∇(rvs[tape_idx.n], f, Val{n′}, p, y, ȳ, xs...) :
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
        @assert typeof(tape[end].value) <: Number
        ȳ = one(tape[end].value)
        rvs = preprocess(forward, tape, ȳ, f, x...)
        return map(n->∇(forward, Val{n}, rvs, tape, ȳ, f, x...), eachindex(x))
    end
end

@inline ∇(x̄, f, ::Type{Val{N}}, args...) where N = x̄ + ∇(f, Val{N}, args...)

"""
    preprocess(::Function, args...)

Default implementation of preprocess returns an empty Tuple. Individual sensitivity
implementations should add methods specific to their use case. The output is passed
in to `∇` as the 3rd or 4th argument in the new-x̄ and update-x̄ cases respectively.
"""
@inline preprocess(::Any, args...) = nothing



########################## Initialisers for containers ################################

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
