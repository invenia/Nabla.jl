# Custom Sensitivities 

!!! note "Prefer to use ChainRulesCore to define custom sensitivities"
    Nabla supports the use of [ChainRulesCore](http://www.juliadiff.org/ChainRulesCore.jl/stable/) to define custom sensitivities.
    It is preferred to define the custom sensitivities using `ChainRulesCore.rrule` as they will work for many AD systems, not just Nabla.
    **It is also much easier, than the Nabla specific way**.
    These sensitivities can be added in your own package, or for Base/StdLib functions they can be added to [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl/).
    To define custom sensitivities using ChainRulesCore, define a `ChainRulesCore.rrule(f, args...; kwargs...)`.
    See the [ChainRules project's documentation for more information](https://www.juliadiff.org/ChainRulesCore.jl/stable/).
    
    After you define an `rrule`, e.g. for `myfunc(i, j)`, you also need to refresh the list of rule, `ChainRulesOverloadGeneration.refresh_rules()`.

    **If you are defining your custom sensitivities using ChainRulesCore then you do not need to read this page**, and can consider it as documenting a legacy feature.
    
    This page exists to describe how Nabla works, and how sensitivities can be directly defined for Nabla.
    Defining sensitivities this way does not make them accessible to other AD systems, but does let you do things that directly depend on how Nabla works.
    It allows for specific definitions of sensitivities that are only defined for Nabla (which might work differently to more generic definitions defined for all AD).

# Legacy Method

Part of the power of Nabla is its extensibility, specifically in the form of defining
custom sensitivities for functions.
This is accomplished by defining methods for `∇` that specialize on the function for
which you'd like to define sensitivities.

Given a function of the form ``f(x_1, \ldots, x_n)``, we want to be able to compute
``\frac{\partial f}{\partial x_i}`` for all ``i`` of interest as efficiently as possible.
Defining our own sensitivities ``\bar{x}_i`` means that ``f`` will be taken as a "unit,"
and its intermediate operations are not written separately to the tape.
For more details on that, refer to the [Details](@ref Automatic-Differentiation) section
of the documentation.

## Intercepting calls

Nabla's approach to RMAD is based on operator overloading.
Specifically, for each ``x_i`` we wish to differentiate, we need a method for `f` that
accepts a `Node` in position ``i``.
There are two primary ways to go about this: `@explicit_intercepts` and `@unionise`.

### `@explicit_intercepts`

When `f` has already been defined, we can extend it to accept `Node`s using this macro.

```@docs
@explicit_intercepts
```

As a trivial example, take `sin` for scalar values (not matrix sine).
We extend it for `Node`s as

```julia
import Base: sin  # ensure sin can be extended without qualification

@explicit_intercepts sin Tuple{Real}
```

This generates the following code:

```julia
begin
    function sin(##367::Node{<:Real})
        #= REPL[7]:1 =#
        Branch(sin, (##367,), getfield(##367, :tape))
    end
end
```

And so calling `sin` with a `Node` argument will produce a `Branch` that holds information
about the call.

For a nontrivial example, take the `sum` function, which accepts a function argument
that gets mapped over the input prior to reduction by addition, as well as a `dims`
keyword argument that permits summing over a subset of the dimensions of the input.
We want to differentiate with respect to the input array, but not with respect to the
function argument nor the dimension.
(Note that Nabla cannot currently differentiate with respect to keyword arguments.)
We can extend this for `Node`s as

```julia
import Base: sum

@explicit_intercepts(
    sum,
    Tuple{Function, AbstractArray{<:Real}},
    [false, true],
    (dims=:,),
)
```

The signature of the call to `@explicit_intercepts` here may look a bit complex, so let's
break it down.
It's saying that we want to intercept calls to `sum` for methods which accept a `Function`
and an `AbstractArray{<:Real}`, and that we do not want to differentiate with respect to
the function argument (`false`) but do want to differentiate with respect to the array
(`true`).
Furthermore, methods of this form will have the keyword argument `dims`, which defaults
to `:`, and we'd like to make sure we're able to capture that when we intercept.

This macro generates the following code:

```
quote
    function sum(##363::Function, ##364::Node{<:Array}; dims=:)
        #= REPL[2]:1 =#
        Branch(sum, (##363, ##364), getfield(##364, :tape); dims=dims)
    end
end
```

As you can see, it defines a new method for `sum` which has positional arguments of
the given types, with the second extended for `Node`s, as well as the given keyword
arguments.
Notice that we do not accept a `Node` for the function argument; this is by virtue of
using `false` in that position in the call to `@explicit_intercepts`.

### `@unionise`

If `f` has not yet been defined and you know off the bat that you want it to be able to
work with Nabla, you can annotate its definition with `@unionise`.

```@docs
@unionise
```

As a simple example,

```julia
@unionise f(x::Matrix, p::Real) = norm(x, p)
```

For each type constrained argument `xi` in the method definition's signature, `@unionise`
changes the type constraint from `T` to `Union{T, Node{<:T}}`, allowing `f` to work with
`Node`s without needing to define separate methods.
In this example, the macro expands the definition to

```julia
f(x::Union{Matrix, Node{<:Matrix}}, p::Union{Real, Node{<:Real}}) = begin
        #= REPL[9]:1 =#
        norm(x, p)
    end
```

## Defining sensitivities

Now that our function `f` works with `Node`s, we want to define a method for `∇` for each
argument `xi` that we're interested in differentiating.
Thus, for each argument position `i` we care about, we'll define a method of `∇` that
looks like:

```julia
function Nabla.∇(::typeof(f), ::Type{Arg{i}}, _, y, ȳ, x1, ..., xn)
    # Compute x̄i
end
```

The method signature contains all of the information it needs to compute the derivative:

* `f`, the function
* `Arg{i}`, which specifies which of the `xi` we're computing the sensitivity of
* `_` (placeholder, typically unused)
* `y`, the result of `y = f(x1, ..., xn)`
* `ȳ`, the "incoming" sensitivity propagated to this call
* `x1, ..., xn`, the inputs to `f`

A fully worked example is provided in the [Details](@ref Automatic-Differentiation) section
of the documentation.

## Testing sensitivities

In order to ensure correctness for custom sensitivity definitions, we can compare the
results against those computed by the method of finite differences.
The finite differencing itself is implemented in the Julia package
[FDM](https://github.com/invenia/FDM.jl), but Nabla defines and exports functionality
that permits checking results against finite differencing.

The primary workhorse function for this is `check_errs`.

```@docs
check_errs
```
