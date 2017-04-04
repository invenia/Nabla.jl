export Node, Branch, Root, grad

abstract Node{T}

"""
An element at the 'bottom' of the computational graph.

Fields:
val - the value of the node.
dval - the reverse-mode sensitivity of the node.
count - the number of active uses that this Branch has.
"""
type Root{T} <: Node{T}
    val::T
    dval::T
    count::Int
end
Root(val) = Root(val, getzero(val), 0   )


"""
A Node is the middle of the computational graph.

Fields:
val - the value of this node produced in the forward pass.
dval - the reverse-mode sensitivity of this node. Computed by child nodes.
f - the function used to generate this Node.
args - a Tuple of values passed into f to generate value. The types of these may be a
       mixture of Node and other types. Nodes are treated specially.
count - the number of active uses that this Branch has.
"""
type Branch{T} <: Node{T}
    val::T
    dval::T
    f::Function
    args::Tuple
    count::Int
end
function Branch(f::Function, args::Tuple)
    val = f(map(unbox, args)...)
    return Branch(val, getzero(val), f, map(increment!, args), 0)
end


""" Increment the counter variable if it's a Node. """
@inline increment!(x::Node) = (x.count += 1; return x)
@inline increment!(x) = x

""" Decrement the counter variable if it's a Node. """
@inline decrement!(x::Node) = (x.count -= 1; return x)
@inline decrement!(x) = x

""" Simple helper function to remove values from boxes if they're a Node. """
@inline unbox(x::Node) = x.val
@inline unbox(x) = x

""" Update the gradient accumulator if it's a Node. """
@inline accumulate!{T}(x::Node{T}, darg::T) = (x.dval += darg)
@inline accumulate!{T, V}(x::Node{T}, darg::V) = error("Type of val and dval not the same.")
@inline accumulate!(x, darg) = nothing


"""
Perform the backward pass given a node object.

Inputs:
y - the node from which you wish to perform the backwards pass.
init (optional) - initial value for the reverse-mode sensitivities. Useful for testing and
    for evaluating subsets of the computational graph. Defaults to the appropriate 1 value.
"""
function grad{T}(y::Node{T}, init::T)
    y.dval = init
    grad_(y)
end
grad(y::Node) = grad(y, getone(y.val))


""" The workhorse for grad. """
grad_(y) = nothing
function grad_(y::Branch)
    if y.count == 0
        dargs = y.f(y, y.val, y.dval, map(unbox, y.args)...)
        for (arg, darg) in zip(y.args, dargs)
            decrement!(arg)
            accumulate!(arg, darg)
            grad_(arg)
        end
    end
end

# TODO: Add semantic sugar to allow you to write things like ∇(x, y) to obtain the gradient
# of y w.r.t. x cleanly. How would you detect if this isn't a meaningful statement? (ie.
# gradient w.r.t. y is zero? You could just initialise the result to the corresponding zero
# element or something.)
