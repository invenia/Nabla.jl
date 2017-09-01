using MNIST, Nabla

""" Implementation of the Adam optimiser. """
type Adam{T<:AbstractArray}
    α::Float64
    β1::Float64
    β2::Float64
    m::T
    v::T
    β1_acc::Float64
    β2_acc::Float64
    ϵ::Float64
end
function Adam{T<:AbstractArray}(θ0::T, α::Float64, β1::Float64, β2::Float64, ϵ::Float64)
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

@inline softplus(x) = log1p.(exp.(x))
@inline logistic(x) = 1 ./ (1 .+ exp.(-x))

function encode(Wh, Wμ, Wσ, X)
    h_enc = tanh.(Wh * X)
    return Wμ * h_enc, exp.(Wσ * h_enc)
end
decode(Vh, Vf, Z) = logistic.(Vf * tanh.(Vh * Z))
klvae(μ, σ², D) = 0.5 * (sum(σ²) + sum(abs2, μ) - convert(Float64, D) - sum(log.(σ²)))

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

    # Initialise the AdaGrad optimiser.
    α, β1, β2, ϵ = 1e-3, 0.9, 0.999, 1e-8
    optWh, optWμ, optWσ, optVh, optVf = Adam.((Wh, Wμ, Wσ, Vh, Vf), α, β1, β2, ϵ)

    # Iterate to learn the parameters.
    scal, ϵ, elbos = size(xtr, 2) / sz, 1e-12, Vector{Float64}(itrs)
    for itr in 1:itrs

        # Pick the mini batch.
        idx = rand(eachindex(ytr_), sz)
        xtr_batch = view(xtr, :, idx)

        # Box paramters that we wish to differentiate w.r.t.
        Wh_, Wμ_, Wσ_, Vh_, Vf_ = Leaf.(Tape(), (Wh, Wμ, Wσ, Vh, Vf))

        # Compute log prior of paramter values.
        logprior = -0.5 * λ * (sum(abs2, Wh_) + sum(abs2, Wμ_) + sum(abs2, Wσ_) +
            sum(abs2, Vh_) + sum(abs2, Vf_))

        # Encode, sample, decode, estimate expection of log likelihood under q(z).
        loglik = 0.0
        μz, σz = encode(Wh_, Wμ_, Wσ_, xtr_batch)
        for l in 1:L
            z = μz .+ σz .* randn(dz, sz)
            f = decode(Vh_, Vf_, z)
            loglik += sum(xtr_batch .* log.(f .+ ϵ) .+
                          (1 .- xtr_batch) .* log.((1 + ϵ) .- f))
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
        iterate!(Wh, ∇logp[Wh_], optWh)
        iterate!(Wμ, ∇logp[Wμ_], optWμ)
        iterate!(Wσ, ∇logp[Wσ_], optWσ)
        iterate!(Vh, ∇logp[Vh_], optVh)
        iterate!(Vf, ∇logp[Vf_], optVf)
        println("logp is $(logp.val) at iterate $itr. Mean elbo is $(elbos[itr])")
    end

    function reconstruct(x)
        μz, σz = encode(Wh, Wμ, Wσ, x)
        return logistic.(Vf * tanh.(Vh * (μz .+ σz .* randn(dz, size(x, 2)))))
    end
    sample_vae() = logistic.(Vf * tanh.(Vh * randn(dz)))

    return sample_vae, reconstruct, elbos
end
