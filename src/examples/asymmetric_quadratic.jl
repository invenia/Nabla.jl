using Nabla

# Generate the values required to compute a matrix quadratic form.
N = 5
B = randn(5, 5)
A = B.'B + UniformScaling(1e-6)

# Low-level API computation of derivatives.

# Construct some trackable data.
x_, y_ = randn(N), randn(N)
x, y = Leaf.(Tape(), (x_, y_))

# Compute the forward pass.
z = x.'A * y
println("Output of the forward pass is:")
println(z)
println()
println("y.val is $(z.val).")
println()

# Get the reverse tape.
z̄ = ∇(z)
println("Output of reverse-pass is")
println(z̄)
println()

# Index into the reverse tape using x to get the gradient of `y` w.r.t. `x`.
x̄ = z̄[x]
println("Gradient of z w.r.t. x at $x_ is $x̄.")
println()

ȳ = z̄[y]
println("Gradient of z w.r.t. y at $y_ is $ȳ")


# (Current) High-Level API computation of derivatives. I will probably maintain this
# interface and extend is significantly as it is currently rather limited. It just returns
# a function which returns the gradient, which isn't really what you want.

# Define the function to be differentiated. Parameters w.r.t. which we want gradients must
# be arguments. Parameters that we don't want gradients w.r.t. should be passed in via a
# closure.
@unionise f(x::AbstractVector, y::AbstractVector) = x.'A * y

# Compute a function `∇f` which computes the derivative of `f` w.r.t. the inputs.
∇f = ∇(f)

# Compute the derivative of `f` w.r.t. `x` at `x_`. Result is currently a 1-Tuple. Might
# introduce a special case for unary functions where it just returns the result.
(x̄, ȳ) = ∇f(x_, y_)

@assert x̄ == z̄[x]
@assert ȳ == z̄[y]
