using AutoGrad, AutoGrad2, BenchmarkTools

function f1(x)
    tape = Tape()
    xr = Root(x, tape)
    y = xr
    for i in 1:N
        y = y .+ xr
    end
    z = sum(y)
    return z
end

function f(x)
    tape = Tape()
    xr = Root(x, tape)
    y = xr
    for i in 1:N
        y = y .+ xr
    end
    z = sum(y)
    return ∇(z)
end

function h_(x)
    y = x
    for i in 1:N
        y = y .+ x
    end
    return sum(y)
end

function h(x)
    ∇h = grad(h_)
    return ∇h(x)
end

function g(x)
    y = x
    for i in 1:N
        y = y .+ x
    end
    return sum(y)
end

N = 2;
M = 10000;
x = ones(M);
@benchmark f1($x)
@benchmark f($x)
@benchmark h($x)
@benchmark g($x)
