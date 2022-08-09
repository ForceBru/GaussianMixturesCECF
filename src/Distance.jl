"""
$(TYPEDSIGNATURES)

Distance between empirical and theoretical ECFs
for _one_ observation `r_n`.
"""
distance_one_obs(p::AV{<:Real}, mu::AV{<:Real}, sigma::AV{<:Real}, r::Real, b::Real)::Real =
    sum(
        # Don't allocate
        p[k] * normal(r, mu[k], 2b + sigma[k]^2)
        for k ∈ eachindex(p)
    )

"""
$(TYPEDSIGNATURES)

Distance between empirical and theoretical ECFs
for a sample of observations `obs`.
"""
function distance(p::AV{<:Real}, mu::AV{<:Real}, sigma::AV{<:Real}, observations::AV{<:Real}, b::Real)::Real
    idx = eachindex(p)
    
    penalty = sum(
        p[k] * p[h] * normal(mu[k], mu[h], 2b + sigma[k]^2 + sigma[h]^2)
        for k ∈ idx, h ∈ idx
    )

    -2/length(observations) * sum(
        distance_one_obs(p, mu, sigma, r, b) for r ∈ observations
    ) + penalty
end

"""
$(TYPEDSIGNATURES)

Extract mixture parameters (weights, means and standard deviations)
from a single vector of parameters.
"""
function get_mix_params(θ::AV{<:Real})
    @assert length(θ) % 3 == 0

    K = length(θ) ÷ 3 # number of components

    p = @view θ[1:K]
	mu = @view θ[K+1:2K]
	sigma = @view θ[2K+1:end]

    (; p, mu, sigma)
end

"""
$(TYPEDSIGNATURES)

Same as `distance`, but all mixture parameters
come from one vector. Used for optimization.
"""
function distance_one_arg(θ::AV{<:Real}, observations::AV{<:Real}, b::Real)
	p, mu, sigma = get_mix_params(θ)

	distance(p, mu, sigma, observations, b)
end
