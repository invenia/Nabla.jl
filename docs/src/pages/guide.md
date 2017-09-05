# Getting Started

`Nabla.jl` has two interfaces, both of which we expose to the end user. We first provide a minimal working example with the low-level interface, and subsequently show how the high-level interface can be used to achieve similar results. More involved examples can be found [here](https://github.com/invenia/Nabla.jl/tree/master/examples).

## A Toy Problem

Consider the gradient of a vector-quadratic function. The following code snippet constructs such a function, and inputs `x` and `y`.
```@example toy
using Nabla

# Generate some data.
rng, N = MersenneTwister(123456), 2
x, y = randn.(rng, [N, N])
A = randn(rng, N, N)

# Construct a vector-quadratic function in `x` and `y`.
f(x, y) = y.' * (A * x)
f(x, y)
```

Only a small amount of [matrix calculus](https://en.wikipedia.org/wiki/Matrix_calculus) is required to the find the gradient of `f(x, y)` w.r.t. `x` and `y`, which we denote by `∇x` and `∇y` respectively, to be

```@example toy
(∇x, ∇y) = (A.'y, A * x)
```

## High-Level Interface
The high-level interface provides a simple way to "just get the gradients" w.r.t. each argument of `f`:
```@example toy
∇x, ∇y = ∇(f)(x, y)
```
This interface is implemented in `core.jl`, and is a thin wrapper of the low-level interface constructed above. Here, we first use `∇` to get a function which, when evaluated, returns the gradient of `f` w.r.t. each of it's inputs at the values of the inputs provided.

We may provide an optional argument to also return the value `f(x, y)`:
```@example toy
(z, (∇x, ∇y)) = ∇(f, true)(x, y)
```

If the gradient w.r.t. a single argument is all that is required, or a subset of the arguments for an N-ary function, we recommend closing over the arguments which respect to which you do not wish to take gradients. For example, to take the gradient w.r.t. just `x`, one could do the following:
```@example toy
∇(x->f(x, y))(x)
```

Furthermore, indexable containers such as `Dict`s behave sensibly. For example, the following lambda with a `Dict`:
```@example toy
∇(d->f(d[:x], d[:y]))(Dict(:x=>x, :y=>y))
```
or a `Vector`
```@example toy
∇(v->f(v[1], v[2]))([x, y])
```

## Low-Level Interface

We now use `Nabla.jl`'s low-level interface to take the gradient of `f` w.r.t. `x` and `y` at the values of `x` and `y` generated above. We first place `x` and `y` into a `Leaf` container. This enables these variables to be traced by `Nabla.jl`. This can be achieved by first creating a `Tape` object, onto which all computations involving `x` and `y` are recorded, as follows:
```julia
tape = Tape()
x_ = Leaf(tape, x)
y_ = Leaf(tape, y)
```
which can be achieved more concisely using Julia's broadcasting capabilities:
```@example toy
x_, y_ = Leaf.(Tape(), (x, y))
```
Note that it is critical that `x_` and `y_` are constructed using the same `Tape` instance. Currently, `Nabla.jl` will fail silently if this is not the case.
We then simply pass `x_` and `y_` to `f` instead of `x` and `y`, and call `∇` on the result:
```@example toy
z_ = f(x_, y_)
```

We can compute the gradients using `∇`, and access them by indexing the output with `x_` and `y_`:
```@example toy
∇z = ∇(z_)
(∇x, ∇y) = (∇z[x_], ∇z[y_])
```

### Inspecting the forward and reverse tapes

```@example toy
z_.tape
```

```@example toy
∇z
```
