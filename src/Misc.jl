"Normal PDF"
@inline normal(x::Real, mu::Real, var::Real) =
    exp(-(x - mu)^2 / (2var)) / sqrt(2π * var)

"""
$(TYPEDSIGNATURES)

Extract mixture parameters (weights, means and standard deviations)
from a single vector of parameters.
"""
function get_mix_params(θ::AV{<:Real})
    @assert length(θ) % 3 == 0

    K = length(θ) ÷ 3 # number of components

    p, mu, sigma = @views begin
        θ[1:K], θ[K+1:2K], abs.(θ[2K+1:end])
    end

    (; p, mu, sigma)
end
