using MNIST, MyOptimisers

function to1hot(y_::Vector)
    y = Matrix{Int}(10, length(y_))
    for n in eachindex(y_)
        for j in 1:10
            y[j, n] = y_[n] == j - 1 ? 1 : 0
        end
    end
    return y
end

@inline function logistic(x)
    return 1 ./ (1 .+ exp(-x))
end

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
    d0, d1, d2, d3 = size(xtr, 1), 100, 100, size(ytr, 1)
    W1, W2, W3 = 0.1 * randn(d1, d0), 0.1 * randn(d2, d1), 0.1 * randn(d3, d2)
    λ = 1e-3 # I forget what the advice is re. setting different priors for different layers.

    # Initialise the AdaGrad optimiser. (Currently one for each set of parameters, legacy issue
    # with optimisation code, should be sorted out before showing this to anyone...)
    η = 3e-2
    optW1 = AdaGrad(zeros(W1), η)
    optW2 = AdaGrad(zeros(W2), η)
    optW3 = AdaGrad(zeros(W3), η)

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

        # Update the parameters using AdaGrad by indexing into the ∇logp tape.
        iterate!(W1, ∇logp[W1r], optW1)
        iterate!(W2, ∇logp[W2r], optW2)
        iterate!(W3, ∇logp[W3r], optW3)
        println("logp is $(logp.val) at iterate $itr. Mean loglik is $(loglik.val / size(xtr, 2))")
    end
end
