import Base: mapreduce, sum

@eval begin
    @explicit_intercepts(
        mapreduce,
        Tuple{Function, $plustype, AbstractArray{<:∇Scalar}},
        [false, false, true],
        (dims=:,),
    )
    function ∇(
        ::typeof($(kwfname(mapreduce))),
        ::Type{Arg{3}},
        p, y, ȳ, f,
        ::$plustype,
        A::AbstractArray{<:∇Scalar},
        dims=:,
    )
        hasmethod(∇, Tuple{typeof(f), Type{Arg{1}}, ∇Scalar}) ?
            broadcast((An, ȳn)->ȳn * ∇(f, Arg{1}, An), A, ȳ) :
            broadcast((An, ȳn)->ȳn * fmad(f, (An,), Val{1}), A, ȳ)
    end

    @explicit_intercepts(
        sum,
        Tuple{Function, AbstractArray{<:∇Scalar}},
        [false, true],
        (dims=:,),
    )
    @explicit_intercepts(
        sum,
        Tuple{AbstractArray{<:∇Scalar}},
        [true],
        (dims=:,),
    )
    function ∇(
        ::typeof($(kwfname(sum))),
        ::Type{Arg{2}},
        p, y, ȳ,
        f::Function,
        A::AbstractArray{<:∇Scalar},
        dims=:,
    )
        # Just pass through to mapreduce
        return ∇($(kwfname(mapreduce)), Arg{3}, p, y, ȳ, f, Base.add_sum, A, dims)
    end
    function ∇(
        ::typeof($(kwfname(sum))),
        ::Type{Arg{1}},
        p, y, ȳ,
        A::AbstractArray{<:∇Scalar},
        dims=:,
    )
        # Again pass through to mapreduce, using identity as the mapped function
        return ∇($(kwfname(mapreduce)), Arg{3}, p, y, ȳ, identity, Base.add_sum, A, dims)
    end
    # Specialize on sum(abs2, x) as it's a common pattern with a simple derivative
    function ∇(
        ::typeof($(kwfname(sum))),
        ::Type{Arg{2}},
        p, y, ȳ,
        ::typeof(abs2),
        A::AbstractArray{<:∇Scalar},
        dims=:,
    )
        return 2ȳ .* A
    end
end
