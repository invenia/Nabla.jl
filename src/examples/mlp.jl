using MNIST, MyOptimisers, BenchmarkTools
export demo_mlp

function to1hot(y_::Vector)
    y = Matrix{Int}(10, length(y_))
    for n in eachindex(y_)
        for j in 1:10
            y[j, n] = y_[n] == j - 1 ? 1 : 0
        end
    end
    return y
end
accuracy(Y, Ypr) = mean(all(Y .== Ypr, 1))
@inline logistic(x) = 1 ./ (1 .+ exp(-x))

# A simple Multilayer Feedforward Neural Network (Multi-layer Perceptron (MLP)) example for
# classifying the MNIST data set.
function demo_mlp(itrs::Int, sz::Int, η::Float64)

    # Load the data.
    println("Loading data.")
    xtr, ytr_ = traindata()
    xte, yte_ = testdata()

    # Convert to 1-hot label encoding.
    ytr, yte = to1hot(ytr_), to1hot(yte_)

    # Initialise parameters.
    println("Initialising parameters.")
    d0, d1, d2, d3 = size(xtr, 1), 500, 300, size(ytr, 1)
    W1, W2, W3 = 0.1 * randn(d1, d0), 0.1 * randn(d2, d1), 0.1 * randn(d3, d2)
    λ = 1e-3 # I forget what the advice is re. setting different priors for different layers.

    # Initialise the AdaGrad optimiser. (Currently one for each set of parameters, legacy issue
    # with optimisation code, should be sorted out before showing this to anyone...)
    α, β1, β2, ϵ = 1e-3, 0.9, 0.999, 1e-8
    optW1 = Adam(W1, α, β1, β2, ϵ)
    optW2 = Adam(W2, α, β1, β2, ϵ)
    optW3 = Adam(W3, α, β1, β2, ϵ)

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
        W1r, W2r, W3r = Root(W1, tape), Root(W2, tape), Root(W3, tape)

        # Compute log prior of paramter values. Note that we don't want gradients w.r.t. λ,
        # the prior precision, so we don't bother to make it a Node.
        logprior = -0.5 .* λ .* (sumabs2(W1r) + sumabs2(W2r) + sumabs(W3r))

        # Compute new data representation via MLP on a subset of the data.
        f = logistic(W3r * tanh(W2r * tanh(W1r * xtr_batch)))
        f = f ./ sum(f, 1)

        # Compute the log probability of the observations.
        ϵ = 1e-15
        loglik = scal * sum(ytr_batch .* log(f .+ ϵ) .+ (1 .- ytr_batch) .* log((1 + ϵ) .- f))

        # Compute gradient of log-joint w.r.t. each of the parameters.
        logp = loglik .+ logprior
        ∇logp = ∇(logp)

        # Compute ML classification rate for each prediction made this iteration.
        ypr = zeros(d3, sz)
        for n in 1:sz
            ypr[indmax(f.val[:, n]), n] = 1.
        end
        acc = accuracy(ytr_batch, ypr)

        # Update the parameters using AdaGrad by indexing into the ∇logp tape.
        iterate!(W1, ∇logp[W1r], optW1)
        iterate!(W2, ∇logp[W2r], optW2)
        iterate!(W3, ∇logp[W3r], optW3)
        println("logp is $(logp.val) at iterate $itr. Mean loglik is $(loglik.val / size(xtr, 2)). Accuracy is $acc")
    end
end
