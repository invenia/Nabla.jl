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
    "location": "pages/api.html#Nabla.Arg",
    "page": "API",
    "title": "Nabla.Arg",
    "category": "Type",
    "text": "Used to flag which argument is being specified in x̄. \n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.Branch",
    "page": "API",
    "title": "Nabla.Branch",
    "category": "Type",
    "text": "A Branch is a Node with parents (args).\n\nFields: val - the value of this node produced in the forward pass. f - the function used to generate this Node. args - Values indicating which elements in the tape will require updating by this node. tape - The Tape to which this Branch is assigned. pos - the location of this Branch in the tape to which it is assigned.\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.Leaf",
    "page": "API",
    "title": "Nabla.Leaf",
    "category": "Type",
    "text": "An element at the 'bottom' of the computational graph.\n\nFields: val - the value of the node. tape - The Tape to which this Leaf is assigned. pos - the location of this Leaf in the tape to which it is assigned.\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.Node",
    "page": "API",
    "title": "Nabla.Node",
    "category": "Type",
    "text": "Basic unit on the computational graph.\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.Tape",
    "page": "API",
    "title": "Nabla.Tape",
    "category": "Type",
    "text": "A topologically ordered collection of Nodes. \n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.backward_fdm-Tuple{Int64,Vararg{Any,N} where N}",
    "page": "API",
    "title": "Nabla.backward_fdm",
    "category": "Method",
    "text": "backward_fdm(p::Int, ...)\nforward_fdm(p::Int, ...)\ncentral_fdm(p::Int, ...)\n\nConstruct a backward, forward, or central finite-difference method of order p. See fdm for further details.\n\nArguments\n\np::Int: Order of the method.\n\nFurther takes, in the following order, the arguments q, ε, M, and report from fdm.\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.check_errs-Union{Tuple{Any,Union{AbstractArray{#s1,N} where N where #s1<:Real, Real},T,T,Real,Real}, Tuple{Any,Union{AbstractArray{#s1,N} where N where #s1<:Real, Real},T,T,Real}, Tuple{Any,Union{AbstractArray{#s1,N} where N where #s1<:Real, Real},T,T}, Tuple{T}} where T",
    "page": "API",
    "title": "Nabla.check_errs",
    "category": "Method",
    "text": "check_errs(\n    f,\n    ȳ::∇ArrayOrScalar,\n    x::T,\n    v::T,\n    ε_abs::∇Scalar=1e-13,\n    ε_rel::∇Scalar=1e-7\n)::Bool where T\n\nCheck that the difference between finite differencing directional derivative estimation and RMAD directional derivative computation for function f at x in direction v, for both allocating and in-place modes, has absolute and relative errors of ε_abs and ε_rel respectively, when scaled by reverse-mode sensitivity ȳ.\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.fdm-Tuple{Array{#s135,1} where #s135<:Real,Int64}",
    "page": "API",
    "title": "Nabla.fdm",
    "category": "Method",
    "text": "function fdm(\n    grid::Vector{<:∇Scalar},\n    q::Int;\n    ε::∇Scalar=eps(),\n    M::∇Scalar=1e6,\n    report::Bool=false\n)\n\nConstruct a function method(f::Function, x::∇Scalar, h::∇Scalar=ĥ) that takes in a function f, a point x in the domain of f, and optionally a step size h, and estimates the q'th order derivative of f at x with a length(grid)'th order finite-difference method.\n\nArguments\n\ngrid::Vector{<:∇Scalar}: Relative spacing of samples of f that are used by the method.   The length of grid determines the order of the method.\nq::Int: Order of the derivative to estimate. q must be strictly less than the order   of the method.\n\nKeywords\n\nε::∇Scalar=eps(): Absolute roundoff error of the function evaluations.\nM::∇Scalar=1e6: Assumed upper bound of f and all its derivatives at x.\nreport::Bool=false: Also return an instance of FDMReport containing information   about the method constructed.\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.preprocess-Tuple{Any,Vararg{Any,N} where N}",
    "page": "API",
    "title": "Nabla.preprocess",
    "category": "Method",
    "text": "preprocess(::Function, args...)\n\nDefault implementation of preprocess returns an empty Tuple. Individual sensitivity implementations should add methods specific to their use case. The output is passed in to ∇ as the 3rd or 4th argument in the new-x̄ and update-x̄ cases respectively.\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.∇",
    "page": "API",
    "title": "Nabla.∇",
    "category": "Function",
    "text": "∇(f::Function)\n\nReturns a function which, when evaluated with arguments that are accepted by f, will return the gradient w.r.t. each of the arguments.\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.∇-Union{Tuple{Nabla.Node{T},T}, Tuple{T}} where T",
    "page": "API",
    "title": "Nabla.∇",
    "category": "Method",
    "text": "∇(y::Node{<:∇Scalar})\n∇(y::Node{T}, ȳ::T) where T\n\nReturn a Tape object which can be indexed using Nodes, each element of which contains the result of multiplying ȳ by the transpose of the Jacobian of the function specified by the Tape object in y. If y is a scalar and ȳ = 1 then this is equivalent to computing the gradient of y w.r.t. each of the elements in the Tape.\n\n∇(f::Function, ::Type{Arg{N}}, p, y, ȳ, x...)\n∇(x̄, f::Function, ::Type{Arg{N}}, p, y, ȳ, x...)\n\nTo implement a new reverse-mode sensitivity for the N^{th} argument of function f. p is the output of preprocess. x1, x2,... are the inputs to the function, y is its output and ȳ the reverse-mode sensitivity of y.\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.@explicit_intercepts-Tuple{Union{Expr, Symbol},Expr,Expr}",
    "page": "API",
    "title": "Nabla.@explicit_intercepts",
    "category": "Macro",
    "text": "@explicit_intercepts(f::Symbol, type_tuple::Expr, is_node::Expr)\n@explicit_intercepts(f::Symbol, type_tuple::Expr)\n\nCreate a collection of methods which intecept the function calls to f in which at least one argument is a Node. Types of arguments are specified by the type tuple expression in type_tuple. If there are arguments which are not differentiable, they can be specified by providing a boolean vector is_node which indicates those arguments that are differentiable with true values and those which are not as false.\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.@union_intercepts-Tuple{Symbol,Expr,Expr}",
    "page": "API",
    "title": "Nabla.@union_intercepts",
    "category": "Macro",
    "text": "@union_intercepts f type_tuple invoke_type_tuple\n\nInterception strategy based on adding a method to f which accepts the union of each of the types specified by type_tuple. If none of the arguments are Nodes then the method of f specified by invoke_type_tuple is invoked.\n\n\n\n"
},

{
    "location": "pages/api.html#Nabla.@unionise-Tuple{Any}",
    "page": "API",
    "title": "Nabla.@unionise",
    "category": "Macro",
    "text": "@unionise code\n\nTransform code such that each function definition accepts Node objects as arguments, without effecting dispatch in other ways.\n\n\n\n"
},

{
    "location": "pages/api.html#",
    "page": "API",
    "title": "API",
    "category": "page",
    "text": "Modules = [Nabla]\nPrivate = false"
},

]}
