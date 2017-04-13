using MNIST, MyOptimisers, BenchmarkTools, PyPlot, PyCall
export demo_vae

@inline softplus(x) = log1p(exp(x))

function encode(Wh, Wμ, Wσ, X)
    h_enc = tanh(Wh * X)
    return Wμ * h_enc, exp(Wσ * h_enc)
end
decode(Vh, Vf, Z) = logistic(Vf * tanh(Vh * Z))
klvae(μ, σ², D) = 0.5 * (sum(σ²) + sumabs2(μ) - convert(Float64, D) - sum(log(σ²)))

# Generate and learn in a vanilla vae.
function demo_vae(itrs::Int, sz::Int, L::Int) 

    # Load the data and normalise.
    xtr, ytr_ = traindata()
    xtr ./= 256

    # Initialise parameters.
    dx, dz, dh_enc, dh_dec = size(xtr, 1), 10, 500, 500 
    Wh, Wμ, Wσ = 0.1 * randn(dh_enc, dx), 0.1 * randn(dz, dh_enc), 0.1 * randn(dz, dh_enc)
    Vh, Vf = 0.1 * randn(dh_dec, dz), 0.1 * randn(dx, dh_dec)
    λ = 1e-3

    # Initialise the AdaGrad optimiser. (Currently one for each set of parameters, legacy issue
    # with optimisation code, should be sorted out before showing this to anyone...)
    # optWh, optWμ, optWσ = AdaGrad(zeros(Wh), η), AdaGrad(zeros(Wμ), η), AdaGrad(zeros(Wσ), η)
    # optVh, optVf = AdaGrad(zeros(Vh), η), AdaGrad(zeros(Vf), η)
    α, β1, β2, ϵ = 1e-3, 0.9, 0.999, 1e-8
    optWh, optWμ, optWσ, optVh, optVf = map(x->Adam(x, α, β1, β2, ϵ), (Wh, Wμ, Wσ, Vh, Vf))

    # Iterate to learn the parameters.
    scal, ϵ, elbos = size(xtr, 2) / sz, 1e-12, Vector{Float64}(itrs)
    for itr in 1:itrs

        # Pick the mini batch.
        idx = rand(eachindex(ytr_), sz)
        xtr_batch = view(xtr, :, idx)

        # Initialise computational graph.
        tape = Tape()
        Whr, Wμr, Wσr, Vhr, Vfr = map(x->Root(x, tape), (Wh, Wμ, Wσ, Vh, Vf))

        # Compute log prior of paramter values.
        logprior = -0.5 .* λ .* (sumabs2(Whr) + sumabs2(Wμr) + sumabs(Wσr) +
            sumabs2(Vhr) + sumabs2(Vfr))

        # Encode, sample, decode, estimate expection of log likelihood under q(z).
        loglik = 0.0
        μz, σz = encode(Whr, Wμr, Wσr, xtr_batch)
        for l in 1:L
            z = μz .+ σz .* randn(dz, sz)
            f = decode(Vhr, Vfr, z)
            loglik += sum(xtr_batch .* log(f .+ ϵ) .+ (1 .- xtr_batch) .* log((1 + ϵ) .- f))
        end
        loglik /= convert(Float64, L)

        # Compute the kl divergence between local posteriors and prior.
        klqp = klvae(μz, σz .* σz, dz)

        # Compute gradient of log-joint w.r.t. each of the parameters.
        elbo = loglik - klqp
        elbos[itr] = elbo.val / sz
        logp = scal * elbo + logprior
        ∇logp = ∇(logp)

        # Update the parameters using AdaGrad by indexing into the ∇logp tape.
        iterate!(Wh, ∇logp[Whr], optWh)
        iterate!(Wμ, ∇logp[Wμr], optWμ)
        iterate!(Wσ, ∇logp[Wσr], optWσ)
        iterate!(Vh, ∇logp[Vhr], optVh)
        iterate!(Vf, ∇logp[Vfr], optVf)
        println("logp is $(logp.val) at iterate $itr. Mean elbo is $(elbos[itr])")
    end

    function reconstruct(x)
        μz, σz = encode(Wh, Wμ, Wσ, x)
        return logistic(Vf * tanh(Vh * (μz .+ σz .* randn(dz, size(x, 2)))))
    end
    sample_vae() = logistic(Vf * tanh(Vh * randn(dz)))

    return sample_vae, reconstruct, elbos
end
