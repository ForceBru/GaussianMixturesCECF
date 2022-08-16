function distance_constant(observations::AV{<:Real}, b::Real)::Real
    sum(
        normal(observations[n], observations[m], 2 * b^2)
        for n in eachindex(observations), m in eachindex(observations)
    ) / (length(observations)^2)
end

"""
$TYPEDSIGNATURES

Distance between theoretical ECF and the Kernel CF Estimator (KCFE),
which is the CF of a simple kernel density estimator with window size `b>=0`.

- The theoretical CDF is the model whose parameters we're estimating
- The KCFE is a dampened version of the empirical CF, where
the greater the window size `b`, the greater the dampening,
similar to density smoothing in regular KDE.

## Bias-variance tradeoff

- Greater `b` => greater dampening of the empirical CF => greater bias.
- Lower `b` (maybe zero) => more rough empirical CF =>
more overfitting => greater variance.
"""
function distance(
    p::AV{<:Real}, mu::AV{<:Real}, sigma::AV{<:Real}, observations::AV{<:Real}, b::Real,
    constant::Real=distance_constant(observations, b)
)::Real
    N = length(observations)

    loss = -2/N  * sum(
        p[k] * normal(observations[n], mu[k], b^2 + sigma[k]^2)
        for n in eachindex(observations), k in eachindex(p)
    )
    penalty = sum(
        p[j] * p[k] * normal(mu[j], mu[k], sigma[k]^2 + sigma[j]^2)
        for j in eachindex(p), k in eachindex(p)
    )
    
    loss + penalty + constant
end

"""
$TYPEDSIGNATURES

Same as `distance`, but all mixture parameters
come from one vector. Used for optimization.
"""
function distance_one_arg(
    θ::AV{<:Real}, observations::AV{<:Real}, b::Real,
    constant::Real=distance_constant(observations, b)
)
	p, mu, sigma = get_mix_params(θ)

	distance(p, mu, sigma, observations, b, constant)
end
