var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#Nabla-1",
    "page": "Home",
    "title": "Nabla",
    "category": "section",
    "text": "(Image: Build Status) (Image: Windows Build status) (Image: codecov.io) (Image: Stable Docs) (Image: Latest Docs)"
},

{
    "location": "index.html#Getting-Started-1",
    "page": "Home",
    "title": "Getting Started",
    "category": "section",
    "text": "Nabla.jl has two interfaces, both of which we expose to the end user. We first provide a minimal working example with the high-level interface, and subsequently show how the low-level interface can be used to achieve similar results. More involved examples can be found here."
},

{
    "location": "index.html#A-Toy-Problem-1",
    "page": "Home",
    "title": "A Toy Problem",
    "category": "section",
    "text": "Consider the gradient of a vector-quadratic function. The following code snippet constructs such a function, and inputs x and y.using Nabla\n\n# Generate some data.\nrng, N = MersenneTwister(123456), 2\nx, y = randn.(rng, [N, N])\nA = randn(rng, N, N)\n\n# Construct a vector-quadratic function in `x` and `y`.\nf(x, y) = y\' * (A * x)\nf(x, y)Only a small amount of matrix calculus is required to the find the gradient of f(x, y) w.r.t. x and y, which we denote by ∇x and ∇y respectively, to be(∇x, ∇y) = (A\'y, A * x)"
},

{
    "location": "index.html#High-Level-Interface-1",
    "page": "Home",
    "title": "High-Level Interface",
    "category": "section",
    "text": "The high-level interface provides a simple way to \"just get the gradients\" w.r.t. each argument of f:∇x, ∇y = ∇(f)(x, y)This interface is implemented in core.jl, and is a thin wrapper of the low-level interface constructed above. Here, we first use ∇ to get a function which, when evaluated, returns the gradient of f w.r.t. each of it\'s inputs at the values of the inputs provided.We may provide an optional argument to also return the value f(x, y):(z, (∇x, ∇y)) = ∇(f, true)(x, y)If the gradient w.r.t. a single argument is all that is required, or a subset of the arguments for an N-ary function, we recommend closing over the arguments which respect to which you do not wish to take gradients. For example, to take the gradient w.r.t. just x, one could do the following:∇(x->f(x, y))(x)Note that this returns a 1-tuple containing the result, not the result itself!Furthermore, indexable containers such as Dicts behave sensibly. For example, the following lambda with a Dict:∇(d->f(d[:x], d[:y]))(Dict(:x=>x, :y=>y))or a Vector:∇(v->f(v[1], v[2]))([x, y])The methods considered so far have been completely generically typed. If one wishes to use methods whose argument types are restricted then one must surround the definition of the method in the @unionise macro. For example, if only a single definition is required:@unionise g(x::Real) = ...Alternatively, if multiple methods / functions are to be defined, the following format is recommended:@unionise begin\ng(x::Real) = ...\ng(x::T, y::T) where T<:Real = ...\nfoo(x) = ... # This definition is unaffected by `@unionise`.\nend@unionise simply changes the method signature to allow each argument to accept the union of the types specified and Nabla.jl\'s internal Node type. This will have no impact on the performance of your code when arguments of the types specified in the definition are provided, so you can safely @unionise code without worrying about potential performance implications."
},

{
    "location": "index.html#Low-Level-Interface-1",
    "page": "Home",
    "title": "Low-Level Interface",
    "category": "section",
    "text": "We now use Nabla.jl\'s low-level interface to take the gradient of f w.r.t. x and y at the values of x and y generated above. We first place x and y into a Leaf container. This enables these variables to be traced by Nabla.jl. This can be achieved by first creating a Tape object, onto which all computations involving x and y are recorded, as follows:tape = Tape()\nx_ = Leaf(tape, x)\ny_ = Leaf(tape, y)which can be achieved more concisely using Julia\'s broadcasting capabilities:x_, y_ = Leaf.(Tape(), (x, y))Note that it is critical that x_ and y_ are constructed using the same Tape instance. Currently, Nabla.jl will fail silently if this is not the case. We then simply pass x_ and y_ to f instead of x and y:z_ = f(x_, y_)We can compute the gradients of z_ w.r.t. x_ and y_ using ∇, and access them by indexing the output with x_ and y_:∇z = ∇(z_)\n(∇x, ∇y) = (∇z[x_], ∇z[y_])"
},

{
    "location": "index.html#Gotchas-and-Best-Practice-1",
    "page": "Home",
    "title": "Gotchas and Best Practice",
    "category": "section",
    "text": "Nabla.jl does not currently have complete coverage of the entire standard library due to finite resources and competing priorities. Particularly notable omissions are the subtypes of Factorization objects and all in-place functions. These are both issues which will be resolved in the future.\nThe usual RMAD gotcha applies: due to the need to record each of the operations performed in the execution of a function for use in efficient gradient computation, the memory requirement of a programme scales approximately linearly in the length of the programme. Although, due to our use of a dynamically constructed computation graph, we support all forms of control flow, long for / while loops should be performed with care, so as to avoid running out of memory.\nIn a similar vein, develop a (strong) preference for higher-order functions and linear algebra over for-loops; Nabla.jl has optimisations targetting Julia\'s higher-order functions (broadcast, mapreduce and friends), and consequently loop-fusion / \"dot-syntax\", and linear algebra operations which should be made use of where possible."
},

{
    "location": "pages/api.html#Nabla.Arg",
    "page": "API",
    "title": "Nabla.Arg",
    "category": "type",
    "text": "Used to flag which argument is being specified in x̄. \n\n\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.Branch",
    "page": "API",
    "title": "Nabla.Branch",
    "category": "type",
    "text": "A Branch is a Node with parents (args).\n\nFields: val - the value of this node produced in the forward pass. f - the function used to generate this Node. args - Values indicating which elements in the tape will require updating by this node. tape - The Tape to which this Branch is assigned. pos - the location of this Branch in the tape to which it is assigned.\n\n\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.Leaf",
    "page": "API",
    "title": "Nabla.Leaf",
    "category": "type",
    "text": "An element at the \'bottom\' of the computational graph.\n\nFields: val - the value of the node. tape - The Tape to which this Leaf is assigned. pos - the location of this Leaf in the tape to which it is assigned.\n\n\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.Node",
    "page": "API",
    "title": "Nabla.Node",
    "category": "type",
    "text": "Basic unit on the computational graph.\n\n\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.Tape",
    "page": "API",
    "title": "Nabla.Tape",
    "category": "type",
    "text": "A topologically ordered collection of Nodes. \n\n\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.check_errs-Union{Tuple{T}, Tuple{Any,Union{Number, AbstractArray{#s12,N} where N where #s12<:Number},T,T}, Tuple{Any,Union{Number, AbstractArray{#s12,N} where N where #s12<:Number},T,T,Number}, Tuple{Any,Union{Number, AbstractArray{#s12,N} where N where #s12<:Number},T,T,Number,Number}} where T",
    "page": "API",
    "title": "Nabla.check_errs",
    "category": "method",
    "text": "check_errs(\n    f,\n    ȳ::∇ArrayOrScalar,\n    x::T,\n    v::T,\n    ε_abs::∇Scalar=1e-10,\n    ε_rel::∇Scalar=1e-7\n)::Bool where T\n\nCheck that the difference between finite differencing directional derivative estimation and RMAD directional derivative computation for function f at x in direction v, for both allocating and in-place modes, has absolute and relative errors of ε_abs and ε_rel respectively, when scaled by reverse-mode sensitivity ȳ.\n\n\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.domain1-Union{Tuple{T}, Tuple{Function,Function,Array{T,1}}} where T",
    "page": "API",
    "title": "Nabla.domain1",
    "category": "method",
    "text": "domain1{T}(in_domain::Function, measure::Function, points::Vector{T})\ndomain1(f::Function)\n\nAttempt to find a domain for a unary, scalar function f.\n\nArguments\n\nin_domain::Function: Function that takes a single argument x and returns whether x   argument is in f\'s domain.\nmeasure::Function: Function that measures the size of a set of points for f.\npoints::Vector{T}: Ordered set of test points to construct the domain from.\n\n\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.domain2-Tuple{Function}",
    "page": "API",
    "title": "Nabla.domain2",
    "category": "method",
    "text": "domain2(f::Function)\n\nAttempt to find a rectangular domain for a binary, scalar function f.\n\n\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.in_domain-Tuple{Function,Vararg{Float64,N} where N}",
    "page": "API",
    "title": "Nabla.in_domain",
    "category": "method",
    "text": "in_domain(f::Function, x::Float64...)\n\nCheck whether an input x is in a scalar, real function f\'s domain.\n\n\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.preprocess-Tuple{Any,Vararg{Any,N} where N}",
    "page": "API",
    "title": "Nabla.preprocess",
    "category": "method",
    "text": "preprocess(::Function, args...)\n\nDefault implementation of preprocess returns an empty Tuple. Individual sensitivity implementations should add methods specific to their use case. The output is passed in to ∇ as the 3rd or 4th argument in the new-x̄ and update-x̄ cases respectively.\n\n\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.∇",
    "page": "API",
    "title": "Nabla.∇",
    "category": "function",
    "text": "∇(f; get_output::Bool=false)\n\nReturns a function which, when evaluated with arguments that are accepted by f, will return the gradient w.r.t. each of the arguments.\n\n\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.∇-Tuple{Node,Any}",
    "page": "API",
    "title": "Nabla.∇",
    "category": "method",
    "text": "∇(y::Node{<:∇Scalar})\n∇(y::Node{T}, ȳ::T) where T\n\nReturn a Tape object which can be indexed using Nodes, each element of which contains the result of multiplying ȳ by the transpose of the Jacobian of the function specified by the Tape object in y. If y is a scalar and ȳ = 1 then this is equivalent to computing the gradient of y w.r.t. each of the elements in the Tape.\n\n∇(f::Function, ::Type{Arg{N}}, p, y, ȳ, x...)\n∇(x̄, f::Function, ::Type{Arg{N}}, p, y, ȳ, x...)\n\nTo implement a new reverse-mode sensitivity for the N^{th} argument of function f. p is the output of preprocess. x1, x2,... are the inputs to the function, y is its output and ȳ the reverse-mode sensitivity of y.\n\n\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.@explicit_intercepts",
    "page": "API",
    "title": "Nabla.@explicit_intercepts",
    "category": "macro",
    "text": "@explicit_intercepts(f::Symbol, type_tuple::Expr, is_node::Expr[, kwargs::Expr])\n@explicit_intercepts(f::Symbol, type_tuple::Expr)\n\nCreate a collection of methods which intecept the function calls to f in which at least one argument is a Node. Types of arguments are specified by the type tuple expression in type_tuple. If there are arguments which are not differentiable, they can be specified by providing a boolean vector is_node which indicates those arguments that are differentiable with true values and those which are not as false. Keyword arguments to add to the function signature can be specified in kwargs, which must be a NamedTuple.\n\n\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.@union_intercepts",
    "page": "API",
    "title": "Nabla.@union_intercepts",
    "category": "macro",
    "text": "@union_intercepts f type_tuple invoke_type_tuple [kwargs]\n\nInterception strategy based on adding a method to f which accepts the union of each of the types specified by type_tuple. If none of the arguments are Nodes then the method of f specified by invoke_type_tuple is invoked. If applicable, keyword arguments should be provided as a NamedTuple and be added to the generated function\'s signature.\n\n\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.@unionise-Tuple{Any}",
    "page": "API",
    "title": "Nabla.@unionise",
    "category": "macro",
    "text": "@unionise code\n\nTransform code such that each function definition accepts Node objects as arguments, without effecting dispatch in other ways.\n\n\n\n\n\n"
},

{
    "location": "pages/api.html#",
    "page": "API",
    "title": "API",
    "category": "page",
    "text": "Modules = [Nabla]\nPrivate = false"
},

{
    "location": "pages/custom.html#",
    "page": "Custom Sensitivities",
    "title": "Custom Sensitivities",
    "category": "page",
    "text": ""
},

{
    "location": "pages/custom.html#Custom-Sensitivities-1",
    "page": "Custom Sensitivities",
    "title": "Custom Sensitivities",
    "category": "section",
    "text": "Coming soon... (you can already add your own sensitivities easily, we just haven\'t written the documentation yet)."
},

]}
