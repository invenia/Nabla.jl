# Let the user get the `size` and `length` of `Node`s.
Base.size(x::Node, dims...) = size(x.val, dims...)
Base.length(x::Node) = length(x.val)
