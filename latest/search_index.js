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
    "text": "The high-level interface provides a simple way to \"just get the gradients\" w.r.t. each argument of f:∇x, ∇y = ∇(f)(x, y)This interface is implemented in core.jl, and is a thin wrapper of the low-level interface constructed above. Here, we first use ∇ to get a function which, when evaluated, returns the gradient of f w.r.t. each of it\'s inputs at the values of the inputs provided.We may provide an optional argument to also return the value f(x, y):(z, (∇x, ∇y)) = ∇(f; get_output=true)(x, y)If the gradient w.r.t. a single argument is all that is required, or a subset of the arguments for an N-ary function, we recommend closing over the arguments which respect to which you do not wish to take gradients. For example, to take the gradient w.r.t. just x, one could do the following:∇(x->f(x, y))(x)Note that this returns a 1-tuple containing the result, not the result itself!Furthermore, indexable containers such as Dicts behave sensibly. For example, the following lambda with a Dict:∇(d->f(d[:x], d[:y]))(Dict(:x=>x, :y=>y))or a Vector:∇(v->f(v[1], v[2]))([x, y])The methods considered so far have been completely generically typed. If one wishes to use methods whose argument types are restricted then one must surround the definition of the method in the @unionise macro. For example, if only a single definition is required:@unionise g(x::Real) = ...Alternatively, if multiple methods / functions are to be defined, the following format is recommended:@unionise begin\ng(x::Real) = ...\ng(x::T, y::T) where T<:Real = ...\nfoo(x) = ... # This definition is unaffected by `@unionise`.\nend@unionise simply changes the method signature to allow each argument to accept the union of the types specified and Nabla.jl\'s internal Node type. This will have no impact on the performance of your code when arguments of the types specified in the definition are provided, so you can safely @unionise code without worrying about potential performance implications."
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
    "location": "pages/api.html#Nabla.∇-Tuple{Any}",
    "page": "API",
    "title": "Nabla.∇",
    "category": "method",
    "text": "∇(f; get_output::Bool=false)\n\nReturns a function which, when evaluated with arguments that are accepted by f, will return the gradient w.r.t. each of the arguments. If get_output is true, the result of calling f on the given arguments is also returned.\n\n\n\n\n\n"
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
    "text": "Part of the power of Nabla is its extensibility, specifically in the form of defining custom sensitivities for functions. This is accomplished by defining methods for ∇ that specialize on the function for which you\'d like to define sensitivities.Given a function of the form f(x_1 ldots x_n), we want to be able to compute fracpartial fpartial x_i for all i of interest as efficiently as possible. Defining our own sensitivities barx_i means that f will be taken as a \"unit,\" and its intermediate operations are not written separately to the tape. For more details on that, refer to the Details section of the documentation."
},

{
    "location": "pages/custom.html#Intercepting-calls-1",
    "page": "Custom Sensitivities",
    "title": "Intercepting calls",
    "category": "section",
    "text": "Nabla\'s approach to RMAD is based on operator overloading. Specifically, for each x_i we wish to differentiate, we need a method for f that accepts a Node in position i. There are two primary ways to go about this: @explicit_intercepts and @unionise."
},

{
    "location": "pages/custom.html#Nabla.@explicit_intercepts",
    "page": "Custom Sensitivities",
    "title": "Nabla.@explicit_intercepts",
    "category": "macro",
    "text": "@explicit_intercepts(f::Symbol, type_tuple::Expr, is_node::Expr[, kwargs::Expr])\n@explicit_intercepts(f::Symbol, type_tuple::Expr)\n\nCreate a collection of methods which intecept the function calls to f in which at least one argument is a Node. Types of arguments are specified by the type tuple expression in type_tuple. If there are arguments which are not differentiable, they can be specified by providing a boolean vector is_node which indicates those arguments that are differentiable with true values and those which are not as false. Keyword arguments to add to the function signature can be specified in kwargs, which must be a NamedTuple.\n\n\n\n\n\n"
},

{
    "location": "pages/custom.html#@explicit_intercepts-1",
    "page": "Custom Sensitivities",
    "title": "@explicit_intercepts",
    "category": "section",
    "text": "When f has already been defined, we can extend it to accept Nodes using this macro.@explicit_interceptsAs a trivial example, take sin for scalar values (not matrix sine). We extend it for Nodes asimport Base: sin  # ensure sin can be extended without qualification\n\n@explicit_intercepts sin Tuple{Real}This generates the following code:begin\n    function sin(##367::Node{<:Real})\n        #= REPL[7]:1 =#\n        Branch(sin, (##367,), getfield(##367, :tape))\n    end\nendAnd so calling sin with a Node argument will produce a Branch that holds information about the call.For a nontrivial example, take the sum function, which accepts a function argument that gets mapped over the input prior to reduction by addition, as well as a dims keyword argument that permits summing over a subset of the dimensions of the input. We want to differentiate with respect to the input array, but not with respect to the function argument nor the dimension. (Note that Nabla cannot currently differentiate with respect to keyword arguments.) We can extend this for Nodes asimport Base: sum\n\n@explicit_intercepts(\n    sum,\n    Tuple{Function, AbstractArray{<:Real}},\n    [false, true],\n    (dims=:,),\n)The signature of the call to @explicit_intercepts here may look a bit complex, so let\'s break it down. It\'s saying that we want to intercept calls to sum for methods which accept a Function and an AbstractArray{<:Real}, and that we do not want to differentiate with respect to the function argument (false) but do want to differentiate with respect to the array (true). Furthermore, methods of this form will have the keyword argument dims, which defaults to :, and we\'d like to make sure we\'re able to capture that when we intercept.This macro generates the following code:quote\n    function sum(##363::Function, ##364::Node{<:Array}; dims=:)\n        #= REPL[2]:1 =#\n        Branch(sum, (##363, ##364), getfield(##364, :tape); dims=dims)\n    end\nendAs you can see, it defines a new method for sum which has positional arguments of the given types, with the second extended for Nodes, as well as the given keyword arguments. Notice that we do not accept a Node for the function argument; this is by virtue of using false in that position in the call to @explicit_intercepts."
},

{
    "location": "pages/custom.html#Nabla.@unionise",
    "page": "Custom Sensitivities",
    "title": "Nabla.@unionise",
    "category": "macro",
    "text": "@unionise code\n\nTransform code such that each function definition accepts Node objects as arguments, without effecting dispatch in other ways.\n\n\n\n\n\n"
},

{
    "location": "pages/custom.html#@unionise-1",
    "page": "Custom Sensitivities",
    "title": "@unionise",
    "category": "section",
    "text": "If f has not yet been defined and you know off the bat that you want it to be able to work with Nabla, you can annotate its definition with @unionise.@unioniseAs a simple example,@unionise f(x::Matrix, p::Real) = norm(x, p)For each type constrained argument xi in the method definition\'s signature, @unionise changes the type constraint from T to Union{T, Node{<:T}}, allowing f to work with Nodes without needing to define separate methods. In this example, the macro expands the definition tof(x::Union{Matrix, Node{<:Matrix}}, p::Union{Real, Node{<:Real}}) = begin\n        #= REPL[9]:1 =#\n        norm(x, p)\n    end"
},

{
    "location": "pages/custom.html#Defining-sensitivities-1",
    "page": "Custom Sensitivities",
    "title": "Defining sensitivities",
    "category": "section",
    "text": "Now that our function f works with Nodes, we want to define a method for ∇ for each argument xi that we\'re interested in differentiating. Thus, for each argument position i we care about, we\'ll define a method of ∇ that looks like:function Nabla.∇(::typeof(f), ::Type{Arg{i}}, _, y, ȳ, x1, ..., xn)\n    # Compute x̄i\nendThe method signature contains all of the information it needs to compute the derivative:f, the function\nArg{i}, which specifies which of the xi we\'re computing the sensitivity of\n_ (placeholder, typically unused)\ny, the result of y = f(x1, ..., xn)\nȳ, the \"incoming\" sensitivity propagated to this call\nx1, ..., xn, the inputs to fA fully worked example is provided in the Details section of the documentation."
},

{
    "location": "pages/custom.html#Nabla.check_errs",
    "page": "Custom Sensitivities",
    "title": "Nabla.check_errs",
    "category": "function",
    "text": "check_errs(\n    f,\n    ȳ::∇ArrayOrScalar,\n    x::T,\n    v::T,\n    ε_abs::∇Scalar=1e-10,\n    ε_rel::∇Scalar=1e-7\n)::Bool where T\n\nCheck that the difference between finite differencing directional derivative estimation and RMAD directional derivative computation for function f at x in direction v, for both allocating and in-place modes, has absolute and relative errors of ε_abs and ε_rel respectively, when scaled by reverse-mode sensitivity ȳ.\n\n\n\n\n\n"
},

{
    "location": "pages/custom.html#Testing-sensitivities-1",
    "page": "Custom Sensitivities",
    "title": "Testing sensitivities",
    "category": "section",
    "text": "In order to ensure correctness for custom sensitivity definitions, we can compare the results against those computed by the method of finite differences. The finite differencing itself is implemented in the Julia package FDM, but Nabla defines and exports functionality that permits checking results against finite differencing.The primary workhorse function for this is check_errs.check_errs"
},

{
    "location": "pages/autodiff.html#",
    "page": "Details",
    "title": "Details",
    "category": "page",
    "text": ""
},

{
    "location": "pages/autodiff.html#Automatic-Differentiation-1",
    "page": "Details",
    "title": "Automatic Differentiation",
    "category": "section",
    "text": "Automatic differentiation, sometimes abbreviated as \"autodiff\" or simply \"AD,\" refers to the process of computing derivatives of arbitrary functions (in the programming sense of the word) in an automated way. There are two primary styles of automatic differentiation: forward mode (FMAD) and reverse mode (RMAD). Nabla\'s implementation is based on the latter."
},

{
    "location": "pages/autodiff.html#What-is-RMAD?-1",
    "page": "Details",
    "title": "What is RMAD?",
    "category": "section",
    "text": "A comprehensive introduction to AD is out of the scope of this document. For that, the reader may be interested in books such as Evaluating Derivatives by Griewank and Walther. To give a sense of how Nabla works, we\'ll briefly give a high-level overview of RMAD.Say you\'re evaluating a function y = f(x) with the goal of computing the derivative of the output with respect to the input, or, in other words, the sensitivity of the output to changes in the input. Pick an arbitrary intermediate step in the computation of f, and suppose it has the form w = g(u v) for some intermediate variables u and v and function g. We denote the derivative of u with respect to the input x as dotu. In FMAD, this is typically the quantity of interest. In RMAD, we want the derivative of the output y with respect to (each element of) the intermediate variable u, which we\'ll denote baru.Giles (2008) shows us that we can compute the sensitivity of y to changes in u and v in reverse mode asbaru = left( fracpartial gpartial u right)^intercal barw quad\nbarv = left( fracpartial gpartial v right)^intercal barwTo arrive at the desired derivative, we start with the identitybary = fracpartial ypartial y = 1then work our way backward through the computation of f, at each step computing the sensitivities (e.g. barw) in terms of the sensitivities of the steps which depend on it.In Nabla\'s implementation of RMAD, we write these intermediate values and the operations that produced them to what\'s called a tape. In literature, the tape in this context is sometimes referred to as a \"Wengert list.\" We do this because, by virtue of working in reverse, we may need to revisit computed values, and we don\'t want to have to do each computation again. At the end, we simply sum up the values we\'ve stored to the tape."
},

{
    "location": "pages/autodiff.html#How-does-Nabla-implement-RMAD?-1",
    "page": "Details",
    "title": "How does Nabla implement RMAD?",
    "category": "section",
    "text": "Take our good friend f from before, but now call it f, since now it\'s a Julia function containing arbitrary code, among which w = g(u, v) is an intermediate step. With Nabla, we compute fracpartial fpartial x as ∇(f)(x). Now we\'ll take a look inside ∇ to see how the concepts of RMAD translate to Julia."
},

{
    "location": "pages/autodiff.html#Computational-graph-1",
    "page": "Details",
    "title": "Computational graph",
    "category": "section",
    "text": "Consider the computational graph of f, which you can visualize as a directed acyclic graph where each node is an intermediate step in the computation. In our example, it might look something like        x        Input\n       ╱ ╲\n      ╱   ╲\n     u     v     Intermediate values computed from x\n      ╲   ╱\n       ╲ ╱\n        w        w = g(u, v)\n        │\n        y        Outputwhere control flow goes from top to bottom.To model the computational graph of a function, Nabla uses what it calls Nodes, and it stores values to a Tape. Node is an abstract type with subtypes Leaf and Branch. A Leaf is a static quantity that wraps an input value and a tape. As its name suggests, it represents a leaf in the computational graph. A Branch is the result of a function call which has been \"intercepted\" by Nabla, in the sense that one or more arguments passed to it is a Node. It holds the value of from evaluating the call, as well as information about its position in the computational graph and about the call itself. Functions which should produce Branches in the computational graph are explicitly extended to do so; this does not happen automatically for each function."
},

{
    "location": "pages/autodiff.html#Forward-pass-1",
    "page": "Details",
    "title": "Forward pass",
    "category": "section",
    "text": "Nabla starts ∇(f)(x) off by creating a Tape to hold values and constructing a Leaf that references the tape and the input x. It then performs what\'s called the forward pass, where it executes f as usual, walking the computational graph from top to bottom, but with the aforementioned Leaf in place of x. As f is executing, each intercepted function call writes a Branch to the tape. The end result is a fully populated tape that will be used in the reverse pass."
},

{
    "location": "pages/autodiff.html#Reverse-pass-1",
    "page": "Details",
    "title": "Reverse pass",
    "category": "section",
    "text": "During the reverse pass, we make another pass over the computational graph of f, but instead of going from top to bottom, we\'re working our way from bottom to top.We start with an empty tape the same length as the one populated in the forward pass, but with a 1 in the last place, corresponding to the identity bary = 1. We then traverse the forward tape, compute the sensitivity for each Branch, and store it in the corresponding position in the reverse tape. This process happens in an internal function called propagate.The computational graph in question may not be linear, which means we may end up needing to \"update\" a value we\'ve already stored to the tape. By the chain rule, this is just a simple sum of the existing value on the tape with the new value."
},

{
    "location": "pages/autodiff.html#Computing-derivatives-1",
    "page": "Details",
    "title": "Computing derivatives",
    "category": "section",
    "text": "As we\'re propagating sensitivities up the graph during the reverse pass, we\'re calling ∇ on each intermediate computation. In the case of f, this means that when computing the sensitivity w̄ for the intermediate variable w, we will call ∇ on g.This is where the real power of Nabla comes into play. In Julia, every function has its own type, which permits defining methods that dispatch on the particular function passed to it. ∇ makes heavy use of this; each custom sensitivity is implemented as a method of ∇. If no specific method for a particular function has been defined, Nabla enters the function and records its operations as though they were part of the outer computation.In our example, if we have no method ∇ specialized on g, calling ∇ on g during the reverse pass will look inside of g and write each individual operation it does to the tape. If g is large and does a lot of stuff, this can end up writing a lot to the tape. Given that the tape holds the value of each step, that means it could end up using a lot of memory.But if we know how to compute the requisite sensitivities already, we can define a method with the signature∇(::typeof(g), ::Type{Arg{i}}, _, y, ȳ, u, v)where:i denotes the ith argument to g (i.e. 1 for u or 2 for v) which dictates whether we\'re computing e.g. ū (1) or v̄ (2),\n_ is a placeholder that can be safely ignored for our purposes,\ny is the value of g(u, v) computed during the forward pass,\nȳ is the \"incoming\" sensitivity (i.e. the sensitivity propagated to the current call by the call in the previous node of the graph), and\nu and v are the arguments to g.We can also tell Nabla how to update an existing tape value with the computed sensitivity by defining a second method of the form∇(x̄, ::typeof(g), ::Type{Arg{i}}, _, y, ȳ, u, v)which effectively computesx̄ += ∇(g, Arg{i}, _, y, ȳ, u, v)Absent a specific method of that form, the += definition above is used literally."
},

{
    "location": "pages/autodiff.html#A-worked-example-1",
    "page": "Details",
    "title": "A worked example",
    "category": "section",
    "text": "So far we\'ve seen a bird\'s eye view of how Nabla works, so to solidify it a bit, let\'s work through a specific example.Let\'s say we want to compute the derivative ofz = xy + sin(x)where x and y (and by extension z) are scalars. The computational graph looks like      x      y\n      │╲     │\n      │ ╲    │\n      │  ╲   │\n      │   ╲  │\n  sin(x)   x*y\n       ╲   ╱\n        ╲ ╱\n         zA bit of basic calculus tells us thatfracpartial zpartial x = cos(x) + y quad\nfracpartial zpartial y = xwhich means that, using the result noted earlier, our reverse mode sensitivities should bebarx = (cos(x) + y) barz quad\nbary = x barzSince we aren\'t dealing with matrices in this case, we can leave off the transpose of the partials."
},

{
    "location": "pages/autodiff.html#Going-through-manually-1",
    "page": "Details",
    "title": "Going through manually",
    "category": "section",
    "text": "Let\'s try defining a tape and doing the forward pass ourselves:julia> using Nabla\n\njulia> t = Tape()  # our forward tape\nTape with 0 elements\n\njulia> x = Leaf(t, randn())\nLeaf{Float64} 0.6791074260357777\n\njulia> y = Leaf(t, randn())\nLeaf{Float64} 0.8284134829000359\n\njulia> z = x*y + sin(x)\nBranch{Float64} 1.1906804805361544 f=+We can now examine the populated tape t to get a glimpse into what Nabla saw as it walked the tree for the forward pass:julia> t\nTape with 5 elements:\n  [1]: Leaf{Float64} 0.6791074260357777\n  [2]: Leaf{Float64} 0.8284134829000359\n  [3]: Branch{Float64} 0.5625817480655771 f=*\n  [4]: Branch{Float64} 0.6280987324705773 f=sin\n  [5]: Branch{Float64} 1.1906804805361544 f=+We can write this out as a series of steps that correspond to the positions in the tape:w_1 = x\nw_2 = y\nw_3 = w_1 w_2\nw_4 = sin(w_1)\nw_5 = w_3 + w_4Now let\'s do the reverse pass. Here we\'re going to be calling some functions that are called internally in Nabla but aren\'t intended to be user-facing; they\'re used here for the sake of explanation. We start by constructing a reverse tape that will be populated in this pass. The second argument here corresponds to our \"seed\" value, which is typically 1, per the identity barz = 1 noted earlier.julia> z̄ = 1.0\n1.0\n\njulia> rt = Nabla.reverse_tape(z, z̄)\nTape with 5 elements:\n  [1]: #undef\n  [2]: #undef\n  [3]: #undef\n  [4]: #undef\n  [5]: 1.0And now we use our forward and reverse tapes to do the reverse pass, propagating the sensitivities up the computational tree:julia> Nabla.propagate(t, rt)\nTape with 5 elements:\n  [1]: 1.6065471361170487\n  [2]: 0.6791074260357777\n  [3]: 1.0\n  [4]: 1.0\n  [5]: 1.0Revisiting the list of steps, applying the reverse mode sensitivity definition to each, we get a new list, which reads from bottom to top:barw_1 =  fracpartial w_4partial w_1 barw_4 + fracpartial w_3partial w_1 barw_3 =  cos(w_1) barw_4 + w_2 barw_3 =  cos(w_1) + w_2 =  cos(x) + y\nbarw_2 = fracpartial w_3partial w_2 barw_3 = w_1 barw_3 = x\nbarw_3 = fracpartial w_5partial w_3 barw_5 = 1\nbarw_4 = fracpartial w_5partial w_4 barw_5 = 1\nbarw_5 = barz = 1This leaves us withbarx = cos(x) + y = 160655 quad bary = x = 067911which looks familiar! Those are the partial derivatives derived earlier (with barz = 1), evaluated at our values of x and y.We can check our work against what Nabla gives us without going through all of this manually:julia> ∇((x, y) -> x*y + sin(x))(0.6791074260357777, 0.8284134829000359)\n(1.6065471361170487, 0.6791074260357777)"
},

{
    "location": "pages/autodiff.html#Defining-a-custom-sensitivity-1",
    "page": "Details",
    "title": "Defining a custom sensitivity",
    "category": "section",
    "text": "Generally speaking, you won\'t need to go through these steps. Instead, if you have expressions for the partial derivatives, as we did above, you can define a custom sensitivity.Start by defining the function:julia> f(x::Real, y::Real) = x*y + sin(x)\nf (generic function with 1 method)Now we need to tell f that we want Nabla to be able to \"intercept\" it in order to produce an explicit branch on f in the overall computational graph. That means that our computational graph from Nabla\'s perspective is simply    x     y\n     ╲   ╱\n    f(x,y)\n       │\n       zWe do this with the @explicit_intercepts macro, which defines methods for f that accept Node arguments.julia> @explicit_intercepts f Tuple{Real, Real}\nf (generic function with 4 methods)\n\njulia> methods(f)\n# 4 methods for generic function \"f\":\n[1] f(x::Real, y::Real) in Main at REPL[18]:1\n[2] f(363::Real, 364::Node{#s1} where #s1<:Real) in Main at REPL[19]:1\n[3] f(365::Node{#s2} where #s2<:Real, 366::Real) in Main at REPL[19]:1\n[4] f(367::Node{#s3} where #s3<:Real, 368::Node{#s4} where #s4<:Real) in Main at REPL[19]:1Now we define our sensitivities for f as methods of ∇:julia> Nabla.∇(::typeof(f), ::Type{Arg{1}}, _, z, z̄, x, y) = (cos(x) + y)*z̄  # x̄\n\njulia> Nabla.∇(::typeof(f), ::Type{Arg{2}}, _, z, z̄, x, y) = x*z̄  # ȳAnd finally, we can call ∇ on f to compute the partial derivatives:julia> ∇(f)(0.6791074260357777, 0.8284134829000359)\n(1.6065471361170487, 0.6791074260357777)This gives us the same result at which we arrived when doing things manually."
},

]}
