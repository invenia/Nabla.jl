
"""
    generate_overload(sig)

Takes a signature tuple type, for a primal function that has an `rrule` and generates
appropriate overloads for Nabla's `Node` types to allow performing AD.
This is the hook function for `ChainRulesCore.on_new_rule(hook, rrule)`.

For example, if `generate_overload` is called on `Tuple{typeof{identity}, Any}`
then approximately the following code is `@eval`ed:

```julia
function Base.identity(x1::Node{<:Any}; kwargs...)
    args = (x1,)
    (primal_val, pullback) = rrule(op, unbox.(args)...; kwargs...)
    tape = get_tape(args)
    branch = Branch(primal_val, op, args, kwargs.data, tape, length(tape) + 1, pullback)
    push!(tape, branch)
    return branch
end

@inline function preprocess(
    op::typeof(identity), y::Branch, ȳ, x1::Union{Any, Node{<:Any}}
)
    pullback = getfield(y, :pullback)
        @assert pullback !== nothing "pullback not set, ..."
    return pullback(ȳ)
end


@inline function ∇(
    op::typeof(identity), ::Type{Arg{N}}, p, ::Any, ::Any, x1::Union{Any, Node{<:Any}};
    kwargs...
) where N
    return p[N + 1]  # skip dself (N==1) and we don't support functors
end
```

The real code evaluated is a little more complex with macro-hygine and handling for
various complicated type-signatures, including multiple arguments.
"""
function generate_overload(sig)
    opT, argTs = Iterators.peel(ExprTools.parameters(sig))
    opT <: Core.Builtin && return false  # can't do operator overloading for builtins

    isabstracttype(opT) || fieldcount(opT) == 0 || return false  # not handling functors
    isempty(argTs) && return false  # we are an operator overloading AD, need operands

    opT isa DataType && nameof(opT.name.module) == :NaNMath  && return false # Don't care about NaNMath

    # Ignore functions that have complex ranges. This may change when Nabla supports complex
    # numbers.
    opT ∈ typeof.((
        SpecialFunctions.hankelh1, SpecialFunctions.hankelh2,
        log1p, rem2pi, mod, atan, rem,
    ))  && return false
    opT <: Type{<:Complex} && return false  # skip complex constructor

    # Ignore these functions because they have better Nabla specific versions.
    opT ∈ typeof.((
        isapprox, size, length, isassigned,
        Base.Broadcast.combine_styles,  #TODO should i keep this?
    )) && return false

    original_signature_def = build_def(sig)
    unionized_signature_def = copy(original_signature_def)
    unionized_signature_def[:args] = unionise_sig.(original_signature_def[:args])

    fdef = quote
        @inline $(preprocess_declaration(unionized_signature_def))
        @inline $(∇_declaration(unionized_signature_def))
        $(overload_declarations!(original_signature_def)...)
    end
    # for debugging uncomment and edit the below to look at the generated code
    # opT <: typeof(identity) && @show fdef
    eval(fdef)
    return true
end

"""
    build_def(sig)

Like `ExprTools.signature` but on a signature type-tuple, not a Method.
For `sig` being a tuple-type representing a methods type signature, this generates a
dictionary that can be passes to `ExprTools.combinedef` to define that function,
Provided that you assign the `:body` key on the dictionary first.

For example:
```julia
julia> Nabla.build_def(Tuple{typeof(identity), Any})
Dict{Symbol, Any} with 2 entries:
  :name => :(op::typeof(identity))
  :args => Expr[:(x1::Any)]

julia> Nabla.build_def(Tuple{typeof(+), Vector{T}, Vector{T}} where T<:Number)
Dict{Symbol, Any} with 3 entries:
  :name        => :(op::typeof(+))
  :args        => Expr[:(x1::Array{var"##T#5492", 1}), :(x2::Array{var"##T#5492", 1})]
  :whereparams => Any[:(var"##T#5492" <: Number)]
```
"""
function build_def(orig_sig)
    sig = _truely_rename_unionall(orig_sig)  # TODO ExprTools possibly should do this for `signature(::Method)`` also
    def = Dict{Symbol, Any}()

    opT = ExprTools.parameters(sig)[1]
    def[:name] = :(op::$opT)

    explicit_tvars = Core.TypeName[]#ExprTools.extract_tvars(sig)
    arg_types = ExprTools.name_of_type.(ExprTools.argument_types(sig))
    arg_names = [Symbol(:x, ii) for ii in eachindex(arg_types)]  #TODO: should we pass the arg_names in?
    def[:args] = Expr.(:(::), arg_names, arg_types)
    def[:whereparams] = ExprTools.where_parameters(sig)

    def = Dict{Symbol, Any}(k => v for (k, v) in def if v !== nothing)  # filter out nonfields.
    return def
end

"""
    overload_declarations!(original_signature_def)

Given a `signature_def` dictionary as returned by [`build_def`](@ref) this returns
the ASTs for the overloads of the primal functions to accept `Nabla.Node`s.
The `signature_def` should *not* have been unionized, as this function will instead generate
1 method for each position a node could be in.
Note: this mutate `signature_def` and so should not be called if others functions also need
to use it after.
"""
function overload_declarations!(signature_def)
    # Our manual macro-hygine is not complete here.
    # the argument names and `op`, `tape` `args`, `kwargs` etc could conflict with
    # where-params. but for sake of outputting readable code we are not gensyming everything
    # chance of conflict seems low as where-params are normally upper-case.
    @assert(signature_def[:name].head == :(::))
    @assert(signature_def[:name].args[1] == :op)

    original_signature_args = signature_def[:args]
    signature_def[:kwargs] = [:(kwargs...)]
    signature_def[:body] = quote
        args = $(_args_tuple(original_signature_args))
        # uncommenting the below to is useful for debugging what rrule is being hit.
        # @show InteractiveUtils.@which rrule(op, unbox.(args)...)
        primal_val, pullback = rrule(op, unbox.(args)...; kwargs...)
        tape = get_tape(args)

        branch = Branch(primal_val, op, args, kwargs.data, tape, length(tape) + 1, pullback)
        push!(tape, branch)
        return branch
    end

    # we need to generate a version of this for each place that an arg could be
    n_args = length(original_signature_args)
    definitions = Expr[]
    for swap_mask in Iterators.product(ntuple(_->(true,false), n_args)...)
        any(swap_mask) || continue  # don't generate if not swapping anything.
        signature_def[:args] = map(swap_mask, original_signature_args) do swap, orig_arg
            if swap
                @assert Meta.isexpr(orig_arg, :(::), 2)
                Expr(:(::), orig_arg.args[1], node_type(orig_arg.args[2]))
            else
                orig_arg
            end
        end
        push!(definitions, ExprTools.combinedef(signature_def))
    end

    return definitions
end


"""
    preprocess_declaration(unionized_signature_def)

Generates AST for overloads for [`Nabla.preprocess`](@ref) that will call the pullback
stored on the `Branch`.
Roughly speaking generated code like:
`preprocess(f::opT, y::Branch, ȳ, xs...)) = y.pullback(ȳ)`
We need the pullback value to use to compute the sensitivies of the primal inputs, that will
be queries by `∇(::opT, ::Type{Arg{N}}, p, y, ȳ, xs...)` where `p` is that pullback value
return by the `preprocess` function.

Note that the `unionised_signature_def` must already have been unionised to accept `Node`s.
"""
function preprocess_declaration(signature_def)
    op = signature_def[:name]
    args = signature_def[:args]
    y = gensym(:y)
    ȳ = gensym(:ȳ)

    # preprocess has a similar definition, signature-wise, to what is in signature_def
    preprocess_def = Dict{Symbol, Any}(
        :name => :preprocess,
        :args => [op, :($y::Branch), ȳ, args...],
        :body => quote
            pullback = getfield($y, :pullback)  # avoid issues with getproperty overloading
            @assert(pullback !== nothing, "pullback not set, probably because different code path used for preprocess vs for ∇. Probably need to delete a defination for ∇")
            return pullback($ȳ)
        end,
    )

    where_params = get(signature_def, :whereparams, nothing)
    if where_params !== nothing
        preprocess_def[:whereparams] = where_params
    end
    return ExprTools.combinedef(preprocess_def)
end

"""
    ∇_declaration(unionised_signature_def)

Generates that AST for the overload of the `∇` function which returns the gradient for
specified arguments.
Basically this generates things like:
`∇(::opT, ::Type{Arg{N}}, p, y, ȳ, xs...) where N = p[N+1]  # Skip dself`
where `p` is the pullback computed by [`preprocess`](@ref)

Note that the `unionised_signature_def` must already have been unionised to accept `Node`s.
"""
function ∇_declaration(signature_def)
    # For readability lets name all the parts, NB: this is being a bit too cute.
    op = signature_def[:name]
    args = signature_def[:args]

    N = gensym(:N)
    p = gensym(:p)
    y = :(::Any)
    ȳ = :(::Any)

    ∇_def = Dict{Symbol, Any}(
        :name => :∇,
        :args => [op, :(::Type{Arg{$N}}), p, y, ȳ, args...],
        :whereparams => [N; get(signature_def, :whereparams, [])],
        :body => quote $p[$N+1] end,  # skip dself
        :kwargs => [:(kwargs...)],
    )
    return ExprTools.combinedef(∇_def)
end


"""
    _args_tuple(arg_exprs)

For `arg_exprs` being a list of arguments expressions from a signature, of a form
such as `[:(x::Int), :(y::Float64), :(z::Vararg)]`, returns a tuple expresion containing all
of them by name; while correctly handling splatting, for things that are `Vararg` typed.
e.g for the prior example `:((x, y, z...))`
"""
function _args_tuple(arg_exprs)
    ret = Expr(:tuple)
    ret.args = map(arg_exprs) do arg
        @assert Meta.isexpr(arg, :(::), 2)
        arg_name, Texpr = arg.args
        if Meta.isexpr(Texpr, :where)  # remove where from `Vararg{T, N} where {T, N}`
            Texpr = Texpr.args[1]
        end
        # Needs to be after removing `where`
        if Meta.isexpr(Texpr, :curly) # remove `{T, N}` from `Vararg{T,N```
            Texpr = Texpr.args[1]
        end
        if Texpr == :Vararg
            return :($arg_name...)
        else
            return arg_name
        end
    end
    return ret
end

"""
    _truely_rename_unionall(@nospecialize(u))

For `u` being a `UnionAll` this replaces every `TypeVar` with  a new one with a `gensym`ed
names. This is useful for manual macro-hygine.

Example:
```
julia> Nabla._truely_rename_unionall(Array{T, N} where {T<:Number, N})
Array{var"##T#2881", var"##N#2880"} where var"##N#2880" where var"##T#2881"<:Number
```

Note that the similar `Base.rename_unionall`, does not `gensym` the names just replaces the
instances with new one with identical names.
"""
function _truely_rename_unionall(@nospecialize(u))
    isa(u,UnionAll) || return u
    body = _truely_rename_unionall(u.body)
    if body === u.body
        body = u
    else
        body = UnionAll(u.var, body)
    end
    var = u.var::TypeVar
    nv = TypeVar(gensym(var.name), var.lb, var.ub)
    return UnionAll(nv, body{nv})
end


# Find a tape, ds might be Nodes or might be something else.
# All nodes should have the same tape, so the first one will do
get_tape(ds) = first(tape(d) for d in ds if d isa Node)
