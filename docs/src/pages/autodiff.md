# Automatic Differentiation

[Automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation),
sometimes abbreviated as "autodiff" or simply "AD," refers to the process of computing
derivatives of arbitrary functions (in the programming sense of the word) in an automated
way.
There are two primary styles of automatic differentiation: _forward mode_ (FMAD) and
_reverse mode_ (RMAD).
Nabla's implementation is based on the latter.

## What is RMAD?

A comprehensive introduction to AD is out of the scope of this document.
For that, the reader may be interested in books such as _Evaluating Derivatives_ by
Griewank and Walther.
To give a sense of how Nabla works, we'll briefly give a high-level overview of RMAD.

Say you're evaluating a function ``y = f(x)`` with the goal of computing the derivative
of the output with respect to the input, or, in other words, the _sensitivity_ of the
output to changes in the input.
Pick an arbitrary intermediate step in the computation of ``f``, and suppose it has the
form ``w = g(u, v)`` for some intermediate variables ``u`` and ``v`` and function ``g``.
We denote the derivative of ``u`` with respect to the input ``x`` as ``\dot{u}``.
In FMAD, this is typically the quantity of interest.
In RMAD, we want the derivative of the output ``y`` with respect to (each element of) the
intermediate variable ``u``, which we'll denote ``\bar{u}``.

[Giles (2008)](https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf) shows us that we
can compute the sensitivity of ``y`` to changes in ``u`` and ``v`` in reverse mode as
```math
\bar{u} = \left( \frac{\partial g}{\partial u} \right)^{\intercal} \bar{w}, \quad
\bar{v} = \left( \frac{\partial g}{\partial v} \right)^{\intercal} \bar{w}
```
To arrive at the desired derivative, we start with the identity
```math
\bar{y} = \frac{\partial y}{\partial y} = 1
```
then work our way backward through the computation of `f`, at each step computing the
sensitivities (e.g. ``\bar{w}``) in terms of the sensitivities of the steps which depend
on it.

In Nabla's implementation of RMAD, we write these intermediate values and the operations
that produced them to what's called a _tape_.
In literature, the tape in this context is sometimes referred to as a "Wengert list."
We do this because, by virtue of working in reverse, we may need to revisit computed
values, and we don't want to have to do each computation again.
At the end, we simply sum up the values we've stored to the tape.

## How does Nabla implement RMAD?

Take our good friend ``f`` from before, but now call it `f`, since now it's a Julia
function containing arbitrary code, among which `w = g(u, v)` is an intermediate step.
With Nabla, we compute ``\frac{\partial f}{\partial x}`` as `∇(f)(x)`.
Now we'll take a look inside `∇` to see how the concepts of RMAD translate to Julia.

### Computational graph

Consider the _computational graph_ of `f`, which you can visualize as a directed acyclic
graph where each node is an intermediate step in the computation.
In our example, it might look something like
```
        x        Input
       ╱ ╲
      ╱   ╲
     u     v     Intermediate values computed from x
      ╲   ╱
       ╲ ╱
        w        w = g(u, v)
        │
        y        Output
```
where control flow goes from top to bottom.

To model the computational graph of a function, Nabla uses what it calls `Node`s, and it
stores values to a `Tape`.
`Node` is an abstract type with subtypes `Leaf` and `Branch`.
A `Leaf` is a static quantity that wraps an input value and a tape.
As its name suggests, it represents a leaf in the computational graph.
A `Branch` is the result of a function call which has been "intercepted" by Nabla, in
the sense that one or more arguments passed to it is a `Node`.
It holds the value of from evaluating the call, as well as information about its position
in the computational graph and about the call itself.
Functions which should produce `Branch`es in the computational graph are explicitly
extended to do so; this does not happen automatically for each function.

### Forward pass

Nabla starts `∇(f)(x)` off by creating a `Tape` to hold values and constructing a
`Leaf` that references the tape and the input `x`.
It then performs what's called the _forward pass_, where it executes `f` as usual, walking
the computational graph from top to bottom, but with the aforementioned `Leaf` in place of
`x`.
As `f` is executing, each intercepted function call writes a `Branch` to the tape.
The end result is a fully populated tape that will be used in the _reverse pass_.

### Reverse pass

During the reverse pass, we make another pass over the computational graph of ``f``, but
instead of going from top to bottom, we're working our way from bottom to top.

We start with an empty tape the same length as the one populated in the forward pass,
but with a 1 in the last place, corresponding to the identity ``\bar{y} = 1``.
We then traverse the forward tape, compute the sensitivity for each `Branch`, and store
it in the corresponding position in the reverse tape.
This process happens in an internal function called `propagate`.

The computational graph in question may not be linear, which means we may end up needing
to "update" a value we've already stored to the tape.
By the chain rule, this is just a simple sum of the existing value on the tape with the
new value.

### Computing derivatives

As we're propagating sensitivities up the graph during the reverse pass, we're calling
`∇` on each intermediate computation.
In the case of `f`, this means that when computing the sensitivity `w̄` for the intermediate
variable `w`, we will call `∇` on `g`.

This is where the real power of Nabla comes into play.
In Julia, every function has its own type, which permits defining methods that dispatch
on the particular function passed to it.
`∇` makes heavy use of this; each custom sensitivity is implemented as a method of `∇`.
If no specific method for a particular function has been defined, Nabla enters the function
and records its operations as though they were part of the outer computation.

In our example, if we have no method `∇` specialized on `g`, calling `∇` on `g` during
the reverse pass will look inside of `g` and write each individual operation it does to
the tape.
If `g` is large and does a lot of stuff, this can end up writing a lot to the tape.
Given that the tape holds the value of each step, that means it could end up using a lot
of memory.

But if we know how to compute the requisite sensitivities already, we can define a method
with the signature
```julia
∇(::typeof(g), ::Type{Arg{i}}, _, y, ȳ, u, v)
```
where:
* `i` denotes the `i`th argument to `g` (i.e. 1 for `u` or 2 for `v`) which dictates whether
  we're computing e.g. `ū` (1) or `v̄` (2),
* `_` is a placeholder that can be safely ignored for our purposes,
* `y` is the value of `g(u, v)` computed during the forward pass,
* `ȳ` is the "incoming" sensitivity (i.e. the sensitivity propagated to the current call
  by the call in the previous node of the graph), and
* `u` and `v` are the arguments to `g`.

We can also tell Nabla how to update an existing tape value with the computed sensitivity
by defining a second method of the form
```julia
∇(x̄, ::typeof(g), ::Type{Arg{i}}, _, y, ȳ, u, v)
```
which effectively computes
```julia
x̄ += ∇(g, Arg{i}, _, y, ȳ, u, v)
```
Absent a specific method of that form, the `+=` definition above is used literally.

## A worked example

So far we've seen a bird's eye view of how Nabla works, so to solidify it a bit, let's
work through a specific example.

Let's say we want to compute the derivative of
```math
z = xy + \sin(x)
```
where `x` and `y` (and by extension `z`) are scalars.
The computational graph looks like
```
      x      y
      │╲     │
      │ ╲    │
      │  ╲   │
      │   ╲  │
  sin(x)   x*y
       ╲   ╱
        ╲ ╱
         z
```
A bit of basic calculus tells us that
```math
\frac{\partial z}{\partial x} = \cos(x) + y, \quad
\frac{\partial z}{\partial y} = x
```
which means that, using the result noted earlier, our reverse mode sensitivities should
be
```math
\bar{x} = (\cos(x) + y) \bar{z}, \quad
\bar{y} = x \bar{z}
```
Since we aren't dealing with matrices in this case, we can leave off the transpose of
the partials.

### Going through manually

Let's try defining a tape and doing the forward pass ourselves:
```julia-repl
julia> using Nabla

julia> t = Tape()  # our forward tape
Tape with 0 elements

julia> x = Leaf(t, randn())
Leaf{Float64} 0.6791074260357777

julia> y = Leaf(t, randn())
Leaf{Float64} 0.8284134829000359

julia> z = x*y + sin(x)
Branch{Float64} 1.1906804805361544 f=+
```
We can now examine the populated tape `t` to get a glimpse into what Nabla saw as it
walked the tree for the forward pass:
```julia-repl
julia> t
Tape with 5 elements:
  [1]: Leaf{Float64} 0.6791074260357777
  [2]: Leaf{Float64} 0.8284134829000359
  [3]: Branch{Float64} 0.5625817480655771 f=*
  [4]: Branch{Float64} 0.6280987324705773 f=sin
  [5]: Branch{Float64} 1.1906804805361544 f=+
```
We can write this out as a series of steps that correspond to the positions in the tape:

1. ``w_1 = x``
2. ``w_2 = y``
3. ``w_3 = w_1 w_2``
4. ``w_4 = \sin(w_1)``
5. ``w_5 = w_3 + w_4``

Now let's do the reverse pass.
Here we're going to be calling some functions that are called internally in Nabla but
aren't intended to be user-facing; they're used here for the sake of explanation.
We start by constructing a reverse tape that will be populated in this pass.
The second argument here corresponds to our "seed" value, which is typically 1, per the
identity ``\bar{z} = 1`` noted earlier.
```julia-repl
julia> z̄ = 1.0
1.0

julia> rt = Nabla.reverse_tape(z, z̄)
Tape with 5 elements:
  [1]: #undef
  [2]: #undef
  [3]: #undef
  [4]: #undef
  [5]: 1.0
```
And now we use our forward and reverse tapes to do the reverse pass, propagating the
sensitivities up the computational tree:
```julia-repl
julia> Nabla.propagate(t, rt)
Tape with 5 elements:
  [1]: 1.6065471361170487
  [2]: 0.6791074260357777
  [3]: 1.0
  [4]: 1.0
  [5]: 1.0
```
Revisiting the list of steps, applying the reverse mode sensitivity definition to each,
we get a new list, which reads from bottom to top:

1. ``\bar{w}_1 =
    \frac{\partial w_4}{\partial w_1} \bar{w}_4 + \frac{\partial w_3}{\partial w_1} \bar{w}_3 =
    \cos(w_1) \bar{w}_4 + w_2 \bar{w}_3 =
    \cos(w_1) + w_2 =
    \cos(x) + y
    ``
2. ``\bar{w}_2 = \frac{\partial w_3}{\partial w_2} \bar{w}_3 = w_1 \bar{w}_3 = x``
3. ``\bar{w}_3 = \frac{\partial w_5}{\partial w_3} \bar{w}_5 = 1``
4. ``\bar{w}_4 = \frac{\partial w_5}{\partial w_4} \bar{w}_5 = 1``
5. ``\bar{w}_5 = \bar{z} = 1``

This leaves us with
```math
\bar{x} = \cos(x) + y = 1.60655, \quad \bar{y} = x = 0.67911
```
which looks familiar!
Those are the partial derivatives derived earlier (with ``\bar{z} = 1``), evaluated at
our values of ``x`` and ``y``.

We can check our work against what Nabla gives us without going through all of this
manually:
```julia-repl
julia> ∇((x, y) -> x*y + sin(x))(0.6791074260357777, 0.8284134829000359)
(1.6065471361170487, 0.6791074260357777)
```

### Defining a custom sensitivity

Generally speaking, you won't need to go through these steps.
Instead, if you have expressions for the partial derivatives, as we did above, you can
define a custom sensitivity.

Start by defining the function:
```julia-repl
julia> f(x::Real, y::Real) = x*y + sin(x)
f (generic function with 1 method)
```
Now we need to tell `f` that we want Nabla to be able to "intercept" it in order to produce
an explicit branch on `f` in the overall computational graph.
That means that our computational graph from Nabla's perspective is simply
```
    x     y
     ╲   ╱
    f(x,y)
       │
       z
```
We do this with the `@explicit_intercepts` macro, which defines methods for `f` that
accept `Node` arguments.
```julia-repl
julia> @explicit_intercepts f Tuple{Real, Real}
f (generic function with 4 methods)

julia> methods(f)
# 4 methods for generic function "f":
[1] f(x::Real, y::Real) in Main at REPL[18]:1
[2] f(363::Real, 364::Node{#s1} where #s1<:Real) in Main at REPL[19]:1
[3] f(365::Node{#s2} where #s2<:Real, 366::Real) in Main at REPL[19]:1
[4] f(367::Node{#s3} where #s3<:Real, 368::Node{#s4} where #s4<:Real) in Main at REPL[19]:1
```
Now we define our sensitivities for `f` as methods of `∇`:
```julia-repl
julia> Nabla.∇(::typeof(f), ::Type{Arg{1}}, _, z, z̄, x, y) = (cos(x) + y)*z̄  # x̄

julia> Nabla.∇(::typeof(f), ::Type{Arg{2}}, _, z, z̄, x, y) = x*z̄  # ȳ
```
And finally, we can call `∇` on `f` to compute the partial derivatives:
```julia-repl
julia> ∇(f)(0.6791074260357777, 0.8284134829000359)
(1.6065471361170487, 0.6791074260357777)
```
This gives us the same result at which we arrived when doing things manually.
