using AutoGrad, AutoGrad2, BenchmarkTools

N = 50
M = 10000
function f1()
    tape = Tape()
    xr = Root(ones(M), tape)
    y = xr
    for i in 1:N
        y = y .+ xr
    end
    return sum(y)
end

function f()
    tape = Tape()
    xr = Root(ones(M), tape)
    y = xr
    for i in 1:N
        y = y .+ xr
    end
    z = sum(y)
    return ∇(z)
end

function h()
    x = ones(M)
    function h_(x)
        y = x
        for i in 1:N
            y = y .+ x
        end
        return sum(y)
    end
    ∇h = grad(h_)
    return ∇h(x)
end

function g()
    y = ones(M)
    for i in 1:N
        y = y .+ x
    end
    return sum(y)
end

x = randn(10000);
@benchmark f1()
@benchmark f()
@benchmark h()
@benchmark g()

# function dsimd(x, y)
#     @simd for n in eachindex(x)
#         @inbounds y[n] += x[n]
#     end
# end

# function d(x, y)
#     for n in eachindex(x)
#         @inbounds y[n] += x[n]
#     end
# end

# function red(x)
#     tmp = 0.
#     for n in eachindex(x)
#         @inbounds tmp += x[n]
#     end
#     return tmp
# end

# function redsimd(x)
#     tmp = 0.
#     @simd for n in eachindex(x)
#         @inbounds tmp += x[n]
#     end
#     return tmp
# end
