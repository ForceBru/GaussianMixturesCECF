module ExpGrad

import ..get_mix_params

import ForwardDiff, Tullio

const AV = AbstractVector{T} where T

normal_pdf(x::Real, mu::Real, var::Real) = exp(-(x - mu)^2 / (2var)) / sqrt(2π * var)

mixture_pdf(x::Real, p::AV{<:Real}, mu::AV{<:Real}, sigma::AV{<:Real}, b::Real) = 
    sum(
        p[k] * normal_pdf(x, mu[k], 2b + sigma[k]^2)
        for k ∈ eachindex(p)
    )

mixture_pdf(x::Real, θ::AV{<:Real}, b::Real) = mixture_pdf(x, get_mix_params(θ)..., b)

penalty(p::AV{<:Real}, mu::AV{<:Real}, sigma::AV{<:Real}, b::Real) =
    sum(
        p[k] * p[h] * normal_pdf(mu[k], mu[h], 2b + sigma[k]^2 + sigma[h]^2)
        for k ∈ eachindex(p), h ∈ eachindex(p)
    )

penalty(θ::AV{<:Real}, b::Real) = penalty(get_mix_params(θ)..., b)

function distance(observations::AV{<:Real}, p::AV{<:Real}, mu::AV{<:Real}, sigma::AV{<:Real}, b::Real)
    # Tullio.@tullio penalty[1] := p[k] * p[h] * normal_pdf(mu[k], mu[h], 2b + sigma[k]^2 + sigma[h]^2) grad=Dual
    Tullio.@tullio tmp[1] := p[k] * normal_pdf(observations[n], mu[k], 2b + sigma[k]^2) grad=Dual

    2π * (-2/length(observations) * sum(tmp) + penalty(p, mu, sigma, b))
end

distance(observations::AV{<:Real}, θ::AV{<:Real}, b::Real) =
    distance(observations, get_mix_params(θ)..., b)

mutable struct GaussianMixture{T<:Real, I<:Integer}
    "Number of mixture components"
    K::I

    """
    Initial guess for optimizer.

    Format: `[weights; means; standard deviations;]`
    """
    θ0::Vector{T}
end

function GaussianMixture(θ0::AV{T}) where T<:Real
    @assert length(θ0) % 3 == 0

    K = length(θ0) ÷ 3

    GaussianMixture{T, typeof(K)}(K, copy(θ0))
end

function fit_cecf!(mix::GaussianMixture, sample::AV{<:Real}; b::Real, lr::Real=1e-3, tol::Real=1e-6)
    K = mix.K

    objective = θ -> distance(sample, θ, b)
    cfg_grad = ForwardDiff.GradientConfig(objective, mix.θ0, ForwardDiff.Chunk{3K}())

    θ = copy(mix.θ0)
    θ_lag = zero(θ)
    itr::Int64 = 0
    grad = similar(θ)
    metrics = fill(Inf, length(θ))
    while maximum(metrics) > tol
        θ_lag .= θ
        # 1. Compute gradient
        ForwardDiff.gradient!(grad, objective, θ, cfg_grad)

        @views begin
            # 2. Exponentiated grad descent w.r.t. weights
            @. θ[1:K] = θ[1:K] * exp(-lr * grad[1:K])
            θ[1:K] ./= sum(θ[1:K])

            # 3. Gradient descent step w.r.t. everything else
            @. θ[K+1:end] -= lr * grad[K+1:end]
        end

        @. metrics = abs(θ - θ_lag)
        if itr % 10_000 == 0
            metric = maximum(metrics)
            @show itr, objective(θ), metric
        end

        itr += 1
    end

    θ
end

end