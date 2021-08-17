function Base.fill(val::Node, dims::Vararg{Union{Integer,AbstractUnitRange}})
    return invoke(fill, Tuple{Node, Vararg{Any}}, val, dims...)
end

function Base.fill(val::Node, dims::Tuple{Vararg{Integer}})
    return invoke(fill, Tuple{Node, Vararg{Any}}, val, dims)
end