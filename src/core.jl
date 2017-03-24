export Node, Branch, Root, grad

abstract Node{T<:Union{AbstractFloat, AbstractArray}}

""" Zero the dval according to the type. """
@inline getzero{T<:AbstractFloat}(val::T) = 0.0
@inline getzero{T<:AbstractArray}(val::T) = zeros(val)

""" Set the dval to one according to the type. """
@inline getone{T<:AbstractFloat}(val::T) = 1.0
@inline getone{T<:AbstractArray}(val::T) = ones(val)

""" Increment the counter variable if it's a Node. """
@inline increment!(x::Node) = (x.count += 1; return x)
@inline increment!(x) = x

""" Decrement the counter variable if it's a Node. """
@inline decrement!(x::Node) = (x.count -= 1; return x)
@inline decrement!(x) = x

""" Simple helper function to remove values from boxes if they're a Node. """
@inline unbox(x::Node) = x.val
@inline unbox(x) = x


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
    function Branch(f::Function, args::Tuple)
        val = f(map(unbox, args)...)
        return new(val, getzero(val), f, map(increment!, args), 0)
    end
end


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
    Root(val::T) = new(val, getzero(val), 0)
end


"""
Perform the backward pass given a node object.

Inputs:
y - the node from which you wish to perform the backwards pass.
"""
grad(y) = nothing
function grad(y::Branch)
    y.dval = getone(y.val)
    grad_(y)
end


""" The workhorse for grad. """
function grad_(y::Branch)
    if y.count == 0
        dargs = y.f(y.dval, y.val, map(unbox, y.args)...)
        for (arg, darg) in zip(y.args, dargs)
            decrement!(arg)
            arg.dval += darg
            grad(arg)
        end
    end
end
