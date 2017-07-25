import Nabla.DiffBase.@differentiable
@differentiable Quadratic begin

# Generate the values required to compute a matrix quadratic form.
N = 5
B = randn(5, 5)
A = B.'B + UniformScaling(1e-6)

# Low-level API computation of derivatives.

# Construct some trackable data.
x_ = randn(N)
x = Leaf(Tape(), x_)

# Compute the forward pass.
y = x.'A * x
println("Output of the forward pass is:")
println(y)
println()
println("y.val is $(y.val).")
println()

# Get the reverse tape.
ȳ = ∇(y)
println("Output of reverse-pass is")
println(ȳ)
println()

# Index into the reverse tape using x to get the gradient of `y` w.r.t. `x`.
x̄ = ȳ[x]
println("Gradient of y w.r.t. x at $(x.val) is $x̄.")
println()


# (Current) High-Level API computation of derivatives. I will probably maintain this
# interface and extend is significantly as it is currently rather limited. It just returns
# a function which returns the gradient, which isn't really what you want.

# Define the function to be differentiated. Parameters w.r.t. which we want gradients must
# be arguments. Parameters that we don't want gradients w.r.t. should be passed in via a
# closure.
f(x) = x.'A * x

# Compute a function `∇f` which computes the derivative of `f` w.r.t. the input.
∇f = ∇(f)

# Compute the derivative of `f` w.r.t. `x` at `x_`. Result is currently a 1-Tuple. Might
# introduce a special case for unary functions where it just returns the result.
x̄ = ∇f(x_)

@assert x̄[1] == ȳ[x]

end
