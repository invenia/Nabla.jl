#using InteractiveUtils

function generate_overload(sig)
    opT, argTs = Iterators.peel(ExprTools.parameters(sig))
    opT <: Core.Builtin && return false  # can't do operater overloading for builtins

    isabstracttype(opT) || fieldcount(opT) == 0 || return false # not handling functors
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


    signature_def = build_def(sig)
    original_signature_args = signature_def[:args]
    signature_def[:args] = unionise_sig.(original_signature_args)

    fdef = quote
        @inline $(preprocess_declaration(signature_def))
        @inline $(∇_declaration(signature_def))
        $(overload_declarations!(signature_def, original_signature_args)...)
    end
    opT <: typeof(svd) && @show fdef
    eval(fdef)
    return true
end

"like `ExprTools.signature` but on a signature type-tuple, not a Method"
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

"this overwrites and ruins `signature_def` for others"
function overload_declarations!(signature_def, original_signature_args)

    # Our macro-hygine is not complete here.
    # the argument names and `op`, `tape` `args`, `kwargs` etc could conflict with
    # where-params. but for sake of outputting readable code we are not gensyming everything
    # chance of conflict seems low as where-params are normally upper-case.
    @assert(signature_def[:name].head == :(::))
    @assert(signature_def[:name].args[1] == :op)


    signature_def[:kwargs] = [:(kwargs...)]
    signature_def[:body] = quote
        args = $(_args_tuple(signature_def[:args]))
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

function preprocess_declaration(signature_def)
    # basically want to generate things like:
    # `preprocess(f::$opT, y::Branch, ȳ, $((arg_sig)...)) = y.pullback(ȳ)`
    # We need the pullback value to use to compute the sensitivies of the inputs

    op = signature_def[:name]
    args = signature_def[:args]
    y = gensym(:y)
    ȳ = gensym(:ȳ)

    # preprocess has a broadly similar definition, signature-wise, to the overload.
    # so we copy it to get whereparams etc
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


function ∇_declaration(signature_def)
    # basically want to generate things like:
    # `∇(::$opT, ::Type{Arg{N}}, p, y, ȳ, xs...) where N = p[N+1]  # Skip dself`
    # We need the pullback value to use to compute the sensitivies of the inputs

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
of them by name; while correctly handling splatting,
e.g for prior example `:((x, y, z...))`
"""
function _args_tuple(arg_exprs)
    ret = Expr(:tuple)
    ret.args = map(arg_exprs) do arg
        @assert Meta.isexpr(arg, :(::), 2)
        arg_name, Texpr = arg.args
        if Texpr == :Vararg || (Meta.isexpr(Texpr, :curly) && Texpr.args[1] == :Vararg)
            return :($arg_name...)
        else
            return arg_name
        end
    end
    return ret
end

"like `Base.rename_unionall`, but actually gensyms the name also, not just a new instance"
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
