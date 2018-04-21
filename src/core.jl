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

# If there are any boxed arguments, trace execution, otherwise proceed as usual.
@generated propagate(ctx::∇Ctx, tape::Tape, f, args...) =
    any(map(x->x<:Box, args)) ? :(_propagate(ctx, tape, f, args...)) : :(f(args...))

# Execute the function `f` with unboxed `args`, keeping track of `f`, `args` and `f(args)`.
function _propagate(ctx::∇Ctx, tape::Tape, f, args...)
    val = f(map(arg->unbox(ctx, arg), args)...)
    args = map(x->x isa Box{<:∇Ctx} ? TapeIdx(x.meta.data[1]) : x, (f, args...))
    push!(tape, Box(ctx, val, (length(tape) + 1, args)))
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

"""
    ∇(::typeof(forward), ::Type{Val{N}}, rvs::Vector{Any}, y::Tape, ȳ, f, x...) where N

Treats the forward-pass of AD like any other function, and compute it's sensitivities.
Crucially, most of the work here is done by `preprocess`, so this method really just
collects the sensitivities that it computes.
"""
∇(::typeof(forward), ::Type{Val{1}}, ::Vector{Any}, ::Tape, ȳ, f, x...) = NaN
∇(::typeof(forward), ::Type{Val{N}}, rvs::Vector{Any}, y::Tape, ȳ, f, x...) where N = rvs[N-1]

"""
    preprocess(::typeof(forward), tape::Tape, rvs::Vector{Any}, f, x::Vararg{Any, N}) where N

Performs a reverse-pass through the `Tape` object constructed by a call to `forward`.
"""
function preprocess(::typeof(forward), tape::Tape, rvs::Vector{Any}, f, x::Vararg{Any, N}) where N
    @assert all(x .=== map(x->x.value, tape[1:N]))
    for n in reverse(eachindex(tape))
        metadata = tape[n].meta.data
        if isassigned(rvs, n) && !isempty(metadata[2])

            # Get data required to compute sensitivities.
            xs = map(x->x isa TapeIdx ? tape[x.n].value : x, metadata[2][2:end])
            f, y, ȳ = metadata[2][1], tape[n].value, rvs[n]

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

# Utility to generate an empty reverse-pass vector.
init_rvs_tape(y::Tape, ȳ) = setindex!(Vector{Any}(undef, length(y)), ȳ, length(y))
function init_rvs_tape(y::Tape)
    @assert typeof(y[end].value) <: Number
    return init_rvs_tape(y, one(y[end].value))
end

# Convenience function to collect derivative w.r.t. all args at once.
@inline ∇(f, p, y, ȳ, x...) = map(n->∇(f, Val{n}, p, y, ȳ, x...), eachindex(x))
∇(::typeof(forward), p::Nothing, y::Tape, ȳ::Vector{Any}, f, x...) =
    ∇(forward, preprocess(forward, y, ȳ, f, x...), y, ȳ, f, x...)

"""
    ∇(f)

Return a function which accepts the same arguments as `f`, but returns the gradient of `f`.
"""
∇(f) = function(x...)
    y = forward(f, x...)
    return ∇(forward, nothing, y, init_rvs_tape(y), f, x...)[2:end]
end

@inline ∇(x̄, f, ::Type{Val{N}}, args...) where N = x̄ + ∇(f, Val{N}, args...)

"""
    preprocess(::Function, args...)

Default implementation of preprocess returns an empty Tuple. Individual sensitivity
implementations should add methods specific to their use case. The output is passed in to
`∇` as the 3rd or 4th argument in the new-x̄ and update-x̄ cases respectively.
"""
@inline preprocess(::Any, args...) = nothing

"""
    ∇primitive(f)

Utility macro to make declaring new primitives simpler. `@∇primitive Base.abs2` does all
the stuff necessary to make it possible to correctly intercept `Base.abs2`. Doesn't define
the sensitivities though.
"""
macro ∇primitive(f)
    return esc(:(@primitive $f(args...) where __CONTEXT__<:∇Ctx =
        propagate(__trace__.context, __trace__.metadata, $f, args...)))
end

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
