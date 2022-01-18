
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
    op::typeof(identity), y::Branch, ȳ, x1::Union{Any, Node{<:Any}}
)
    pullback = getfield(y, :pullback)
        @assert pullback !== nothing "pullback not set, ..."
    return pullback(ȳ)
end


@inline function ∇(
    op::typeof(identity), ::Type{Arg{N}}, p, ::Any, ::Any, x1::Union{Any, Node{<:Any}};
    kwargs...
) where N
    return p[N + 1]  # skip dself (N==1) as we don't support functors
end
```

The real code evaluated is a little more complex with macro-hygiene and handling for
various complicated type-signatures, including multiple arguments.

It does not generate any code for `rrules` for primal functions that Nabla does not support.
See [`should_use_rrule`](@ref) for more details on what rules we do not use.

This function returns true or false as to whether or not code was generated. While this has
no actual effect in itself, it can be useful for checking how many rules Nabla supports.
"""
function generate_overload(sig)
    should_use_rrule(sig) || return false

    original_signature_def = ExprTools.signature(sig; extra_hygiene=true)
    unionized_signature_def = copy(original_signature_def)
    unionized_signature_def[:args] = unionise_sig.(original_signature_def[:args])

    fdef = quote
        @inline $(preprocess_declaration(unionized_signature_def))
        @inline $(∇_declaration(unionized_signature_def))
        $(overload_declarations!(original_signature_def)...)
    end
    eval(fdef)

    return true
end

"""
    should_use_rrule(sig)

Should we make use of the chainrules `rrule` for the primal function with the given
signature tuple type (`sig`).

We do not use rules for:
    - builtin functions
    - functors / closures
    - functions without any positional arguments
    - functions from the `NaNMath` module
    - functions for working with complex numbers.
    - Non-differentiable functions that we define directly on `Node`s better (like `size`)
    - Non-differentiable functions that are never used in practice and that cause a lot of
      compiler invalidations and so cause a large increase in loading time.
    - functions that cause Nabla issues that we don't use.

Finally this excludes function that at time of last update Nabla had its own rules for
because ChainRules didn't support them.
Generally, for this category once they are added to ChainRules, we should change to using
them from there. This requires also deleting the code from Nabla that provides those rules
currently, so that there is no clash.
"""
function should_use_rrule(sig)
    opT, argTs = Iterators.peel(ExprTools.parameters(sig))
    opT <: Core.Builtin && return false  # can't do operator overloading for builtins

    isabstracttype(opT) || fieldcount(opT) == 0 || return false  # not handling functors
    isempty(argTs) && return false  # we are an operator overloading AD, need operands

    # Don't care about NaNMath
    opT isa DataType && nameof(opT.name.module) == :NaNMath  && return false

    # Ignore functions that have complex ranges. This may change when Nabla supports complex
    # numbers.
    opT ∈ typeof.((
        SpecialFunctions.hankelh1, SpecialFunctions.hankelh2,
        log1p, rem2pi, mod, atan, rem,
    ))  && return false
    opT <: Type{<:Complex} && return false  # skip complex constructor

    # Ignore these functions because they have better Nabla specific versions.
    opT ∈ typeof.((
        isapprox, axes, size, length, isassigned, one, zero,
        Base.Broadcast.combine_styles,  #TODO should i keep this?
    )) && return false

    # Ignore these functions because in practice they are never used and defining them cause
    # a ton of compiler invalidations, making loading slow.
    opT ∈ typeof.((
        string, repr, print, println, write, readlines, eachline, Core.print, Core.println,
        isequal, ==, in, haskey,
        isnothing, ismissing, isfile,
        isbitstype, isbits, isabstracttype, isconcretetype,
        startswith, endswith, join, joinpath, normpath, chomp,
        schedule,  # this one is huge, causes over 2500 invalidations
    )) && return false

    # Rules currently implemented directly in Nabla, but that could use ChainRules in future
    sig <: Union{
        Tuple{typeof(+),AbstractArray,LinearAlgebra.UniformScaling},
        Tuple{typeof(+),LinearAlgebra.UniformScaling,AbstractArray},
        Tuple{typeof(/),Number,AbstractArray},
        Tuple{typeof(LinearAlgebra.BLAS.symm),Char,Char,AbstractArray,AbstractArray},
        Tuple{typeof(LinearAlgebra.BLAS.symm),Char,Char,Number,AbstractArray,AbstractArray},
        Tuple{typeof(LinearAlgebra.BLAS.symv),Char,AbstractArray,AbstractArray},
        Tuple{typeof(LinearAlgebra.BLAS.symv),Char,Number,AbstractArray,AbstractArray},
        Tuple{typeof(LinearAlgebra.BLAS.trmm),Char,Char,Char,Char,Number,AbstractArray,AbstractArray},
        Tuple{typeof(LinearAlgebra.BLAS.trmv),Char,Char,Char,AbstractArray,AbstractArray},
        Tuple{typeof(LinearAlgebra.BLAS.trsm),Char,Char,Char,Char,Number,AbstractArray,AbstractArray},
        Tuple{typeof(LinearAlgebra.BLAS.trsv),Char,Char,Char,AbstractArray,AbstractArray},
        Tuple{typeof(Statistics.mean),Function,AbstractArray},
        Tuple{typeof(\),AbstractArray,Number},
        Tuple{typeof(broadcast),Any,Vararg},
        Tuple{typeof(copy),Any},
        Tuple{typeof(float),Any},
        Tuple{typeof(getindex),Ref},
        Tuple{typeof(kron),AbstractArray,AbstractArray},
        Tuple{typeof(map),Function,Vararg},
        Tuple{typeof(mapfoldl),Any,Union{typeof(+), typeof(Base.add_sum)},Union{Number,AbstractArray}},
        Tuple{typeof(mapfoldr),Any,Union{typeof(+), typeof(Base.add_sum)},Union{Number,AbstractArray}},
        Tuple{typeof(mapreduce),Any,Union{typeof(+), typeof(Base.add_sum)},AbstractArray},
        Tuple{typeof(sum),Function,AbstractArray},
        Tuple{typeof(sum),typeof(abs2),AbstractArray},
    } && return false


    # Functions that cause Nabla to have issues and that we don't use
    sig <: Union{
        Tuple{Type{<:Array}, AbstractArray},  # Nabla support for constructors is limitted
    } && return false
    
    opT ∈ typeof.((
        Base.vect,  # currently having an issue with this being defined twice.
                    # TODO: debug why and if ever we need this
    )) && return false

    return true  # no exclusion applies
end
"""
    overload_declarations!(original_signature_def)

Given a `signature_def` dictionary as returned by `ExprTools.signature` this returns
the ASTs for the overloads of the primal functions to accept `Nabla.Node`s.
The `signature_def` should *not* have been unionized, as this function will instead generate
1 method for each position a node could be in.
Note: this mutate `signature_def` and so should not be called if others functions also need
to use it after.
"""
function overload_declarations!(signature_def)
    # Our manual macro-hygiene is not complete here.
    # the argument names and `op`, `tape` `args`, `kwargs` etc could conflict with
    # where-params. but for sake of outputting readable code we are not gensyming everything
    # chance of conflict seems low as where-params are normally upper-case.
    @assert(signature_def[:name].head == :(::))
    @assert(signature_def[:name].args[1] == :op)

    original_signature_args = signature_def[:args]
    signature_def[:kwargs] = [:(kwargs...)]
    signature_def[:body] = quote
        args = $(ExprTools.args_tuple_expr(original_signature_args))
        # uncommenting the below to is useful for debugging what rrule is being hit.
        # @show InteractiveUtils.@which rrule(op, unbox.(args)...)
        primal_val, pullback = rrule(op, unbox.(args)...; kwargs...)
        tape = get_tape(args)

        branch = Branch(primal_val, op, args, kwargs.data, tape, length(tape) + 1, pullback)
        push!(tape, branch)
        return branch
    end

    # we need to generate a version of this for each place that an arg could be a Node
    n_args = length(original_signature_args)
    definitions = Expr[]
    for swap_mask in Iterators.product(ntuple(_->(true, false), n_args)...)
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


# Find a tape, ds might be Nodes or might be something else.
# All nodes should have the same tape, so the first one will do
get_tape(ds) = first(tape(d) for d in ds if d isa Node)
