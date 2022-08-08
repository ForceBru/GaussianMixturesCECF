module ExpGrad

import ..get_mix_params

import ForwardDiff

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

distance(observations::AV{<:Real}, p::AV{<:Real}, mu::AV{<:Real}, sigma::AV{<:Real}, b::Real) =
    2π * (
        penalty(p, mu, sigma, b)
        -2/length(observations) * sum(
            mixture_pdf(x, p, mu, sigma, b)
            for x in observations
        )
    )

distance(observations::AV{<:Real}, θ::AV{<:Real}, b::Real) =
    distance(observations, get_mix_params(θ)..., b)

mutable struct GaussianMixture
    "Number of mixture components"
    K::Integer

    """
    Initial guess for optimizer.

    Format: `[weights; means; standard deviations;]`
    """
    θ0::AV{<:Real}
end

function GaussianMixture(θ0::AV{<:Real})
    @assert length(θ0) % 3 == 0

    K = length(θ0) ÷ 3

    GaussianMixture(K, θ0)
end

"""
Fit the thing w/ exponentiated gradient descent
FIXME: takes many iterations!!

The following takes more than 100'000 iterations (and ~11 sec)

```
julia> import Random

julia> rng = Random.MersenneTwister(42);

julia> data = [0.2*randn(rng, 200); 0.3*randn(rng, 300)];

julia> mix = G.ExpGrad.GaussianMixture([.5, .5, 0,0, 0.03, 0.06]);

julia> @time G.ExpGrad.fit_cecf!(mix, data, b=0.01, lr=1e-2, tol=1e-7)
(itr, objective(θ), metric) = (0, -6.204313248313339, 0.30170437026721425)
(itr, objective(θ), metric) = (10000, -6.220054193034246, 4.733383557775639e-6)
(itr, objective(θ), metric) = (20000, -6.2201499761380195, 4.8395758385222365e-6)
(itr, objective(θ), metric) = (30000, -6.220248726143577, 4.666240635498031e-6)
(itr, objective(θ), metric) = (40000, -6.220339565989376, 4.155679707962268e-6)
(itr, objective(θ), metric) = (50000, -6.220410731193607, 3.347766860506418e-6)
(itr, objective(θ), metric) = (60000, -6.220456271074969, 2.4179164314352963e-6)
(itr, objective(θ), metric) = (70000, -6.22047977348448, 1.5832682859484581e-6)
(itr, objective(θ), metric) = (80000, -6.220489808180939, 9.63726648928187e-7)
(itr, objective(θ), metric) = (90000, -6.220493528168738, 5.595421378457033e-7)
(itr, objective(θ), metric) = (100000, -6.22049478496818, 3.1588606314025824e-7)
(itr, objective(θ), metric) = (110000, -6.220495186368714, 1.7551534775561706e-7)
 10.290548 seconds (6.69 M allocations: 344.473 MiB, 0.60% gc time)
6-element Vector{Float64}:
  0.7983720380273869
  0.20162796197261318
  0.0055596215036916195
 -0.06912247031142632
  0.23191746816469924
  0.4319762236372301
```

`6.69 M allocations: 344.473 MiB` - that's a lot of allocations.
Bet this is `ForwardDiff.gradient!`'s fault.
"""
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