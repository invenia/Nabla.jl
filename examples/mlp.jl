using MNIST, Nabla

""" Implementation of the Adam optimiser. """
mutable struct Adam{T<:AbstractArray}
    α::Float64
    β1::Float64
    β2::Float64
    m::T
    v::T
    β1_acc::Float64
    β2_acc::Float64
    ϵ::Float64
end
function Adam(θ0::AbstractArray, α::Float64, β1::Float64, β2::Float64, ϵ::Float64)
    return Adam(α, β1, β2, zeros(θ0), zeros(θ0), β1, β2, ϵ)
end

function iterate!(θ::AbstractArray{Float64}, ∇θ::AbstractArray{Float64}, opt::Adam)
    m, v, α, β1, β2, ϵ, β1_acc, β2_acc =
        opt.m, opt.v, opt.α, opt.β1, opt.β2, opt.ϵ, opt.β1_acc, opt.β2_acc
    @inbounds for n in eachindex(θ)
        m[n] = β1 * m[n] + (1.0 - β1) * ∇θ[n]
        v[n] = β2 * v[n] + (1.0 - β2) * ∇θ[n]^2
        m̂ = m[n] / (1 - β1_acc)
        v̂ = v[n] / (1 - β2_acc)
        θ[n] = θ[n] + α * m̂ / (sqrt(v̂) + ϵ)
    end
    opt.β1_acc *= β1
    opt.β2_acc *= β2
end

function to1hot(y_::Vector)
    y = Matrix{Int}(undef, 10, length(y_))
    for n in eachindex(y_)
        for j in 1:10
            y[j, n] = y_[n] == j - 1 ? 1 : 0
        end
    end
    return y
end
accuracy(Y, Ypr) = mean(all(Y .== Ypr, dims=1))
@inline logistic(x) = 1 ./ (1 .+ exp.(-x))

"""
    mlp_log_joint(
        X::AbstractMatrix{T} where T<:Real,
        Y::AbstractMatrix{T} where T<:Real,
        W::NTuple{N, T} where T<:AbstractMatrix{V} where V<:Real,
        b::NTuple{N, T} where T<:AbstractVector{V} where V<:Real,
        λ::AbstractFloat)

Compute the log joint probability of the data `(X, Y)` given weights `W`, biases `b` and
precision for the isotropic Gaussian prior on the weights and biases `λ` (equivalent to
quadratic weight decay) and a Bernoulli likelihood (equivalent to the cross-entropy).
"""
function mlp_log_joint(X, Y, W, b, λ)

    # Compute the log prior probability of the weights. We have assumed an isotropic
    # Gaussian prior over all of the parameters. Note that since we are not trying to learn
    # λ (it is fixed) we can neglect the normalising constant Z = 0.5 * log(2πλ).
    log_prior = 0.0
    for n in 1:length(W)
        log_prior += sum(abs2, W[n]) + sum(abs2, b[n])
    end
    log_prior *= -0.5 * λ

    # Compute the output of the MLP.
    f = logistic.(apply_transforms(X, W, b, tanh))
    f = f ./ sum(f, dims=1)

    # Compute the log likelihood of the observations given the outputs of the MLP. We assume
    ϵ = 1e-15
    log_lik = sum(Y .* log.(f .+ ϵ) .+ (1 - Y) .* log.((1 + ϵ) .- f))

    # Return the joint and the predictive distribution over the labels.
    return log_lik + log_prior, f
end

apply_transforms(X, W::Tuple{Vararg{Any, 1}}, b::Tuple{Vararg{Any, 1}}, f) =
    f.(W[1] * X .+ b[1])
apply_transforms(X, W::Tuple{Vararg{Any, N}}, b::Tuple{Vararg{Any, N}}, f) where N =
    f.(W[N] * apply_transforms(X, W[1:N-1], b[1:N-1], f) .+ b[N])

# A simple Multilayer Feedforward Neural Network (Multi-layer Perceptron (MLP)) example for
# classifying the MNIST data set.
function demo_mlp(itrs::Int, sz::Int)

    # Load the data.
    println("Loading data.")
    xtr, ytr_ = traindata()
    xte, yte_ = testdata()

    # Convert to 1-hot label encoding.
    ytr, yte = to1hot(ytr_), to1hot(yte_)

    # Initialise parameters.
    println("Initialising parameters.")
    d0, d1, d2, d3, λ = size(xtr, 1), 500, 300, size(ytr, 1), 1e-3
    W_ = (0.1 * randn(d1, d0), 0.1 * randn(d2, d1), 0.1 * randn(d3, d2))
    b_ = (0.1 * randn(d1), 0.1 * randn(d2), 0.1 * randn(d3))

    # Initialise the Adam optimiser.
    α, β1, β2, ϵ = 1e-3, 0.9, 0.999, 1e-8
    optW, optb = Adam.(W_, α, β1, β2, ϵ), Adam.(b_, α, β1, β2, ϵ)

    # Iterate to learn the parameters.
    println("Starting learning.")
    scal = size(xtr, 2) / sz
    for itr in 1:itrs

        # Pick the mini batch.
        idx = rand(eachindex(ytr_), sz)
        xtr_batch = view(xtr, :, idx)
        ytr_batch = view(ytr, :, idx)

        # Initialise computational graph.
        tape = Tape()
        W, b = Leaf.(tape, W_), Leaf.(tape, b_)

        # Compute the log marginal probability.
        logp, f = mlp_log_joint(xtr_batch, ytr_batch, W, b, λ)
        ∇logp = ∇(logp)

        # Compute most probably classification for each observation in the batch.
        ypr = zeros(d3, sz)
        for n in 1:sz
            ypr[argmax(Nabla.unbox(f)[:, n]), n] = 1.
        end
        acc = accuracy(ytr_batch, ypr)

        # Update the parameters using AdaGrad by indexing into the ∇logp tape.
        iterate!.(W_, getindex.(∇logp, W), optW)
        iterate!.(b_, getindex.(∇logp, b), optb)
        println("logp is $(Nabla.unbox(logp)) at iteration $itr. Mean loglik is ",
            "$(Nabla.unbox(logp) / size(xtr, 2)). Accuracy is $acc")
    end
end

