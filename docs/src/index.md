# Nabla

[![Build Status](https://travis-ci.org/invenia/Nabla.jl.svg?branch=master)](https://travis-ci.org/invenia/Nabla.jl)
[![Windows Build status](https://ci.appveyor.com/api/projects/status/g0gun5dxbkt631am/branch/master?svg=true)](https://ci.appveyor.com/project/iamed2/nabla-jl/branch/master)
[![codecov.io](http://codecov.io/github/invenia/Nabla.jl/coverage.svg?branch=master)](http://codecov.io/github/invenia/Nabla.jl?branch=master)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://invenia.github.io/Nabla.jl/stable)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://invenia.github.io/Nabla.jl/latest)

# Getting Started

`Nabla.jl` has two interfaces, both of which we expose to the end user. We first provide a minimal working example with the high-level interface, and subsequently show how the low-level interface can be used to achieve similar results. More involved examples can be found [here](https://github.com/invenia/Nabla.jl/tree/master/examples).

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
Note that this returns a 1-tuple containing the result, not the result itself!

Furthermore, indexable containers such as `Dict`s behave sensibly. For example, the following lambda with a `Dict`:
```@example toy
∇(d->f(d[:x], d[:y]))(Dict(:x=>x, :y=>y))
```
or a `Vector`:
```@example toy
∇(v->f(v[1], v[2]))([x, y])
```

The methods considered so far have been completely generically typed. If one wishes to use methods whose argument types are restricted then one must surround the definition of the method in the `@unionise` macro. For example, if only a single definition is required:
```julia
@unionise g(x::Real) = ...
```
Alternatively, if multiple methods / functions are to be defined, the following format is recommended:
```julia
@unionise begin
g(x::Real) = ...
g(x::T, y::T) where T<:Real = ...
foo(x) = ... # This definition is unaffected by `@unionise`.
end
```
`@unionise` simply changes the method signature to allow each argument to accept the union of the types specified and `Nabla.jl`'s internal `Node` type. This will have no impact on the performance of your code when arguments of the types specified in the definition are provided, so you can safely `@unionise` code without worrying about potential performance implications.

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
We then simply pass `x_` and `y_` to `f` instead of `x` and `y`:
```@example toy
z_ = f(x_, y_)
```

We can compute the gradients of `z_` w.r.t. `x_` and `y_` using `∇`, and access them by indexing the output with `x_` and `y_`:
```@example toy
∇z = ∇(z_)
(∇x, ∇y) = (∇z[x_], ∇z[y_])
```

## Gotchas and Best Practice
- `Nabla.jl` does not currently have complete coverage of the entire standard library due to finite resources and competing priorities. Particularly notable omissions are the subtypes of `Factorization` objects and all in-place functions. These are both issues which will be resolved in the future.
- The usual RMAD gotcha applies: due to the need to record each of the operations performed in the execution of a function for use in efficient gradient computation, the memory requirement of a programme scales approximately linearly in the length of the programme. Although, due to our use of a dynamically constructed computation graph, we support all forms of control flow, long `for` / `while` loops should be performed with care, so as to avoid running out of memory.
- In a similar vein, develop a (strong) preference for higher-order functions and linear algebra over for-loops; `Nabla.jl` has optimisations targetting Julia's higher-order functions (`broadcast`, `mapreduce` and friends), and consequently loop-fusion / "dot-syntax", and linear algebra operations which should be made use of where possible.
