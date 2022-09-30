abstract type DistanceKind{T<:Real} end

"""
$TYPEDEF

Weighted distance between mixture characteristic function (CF)
and empirical characteristic function (ECF) as defined in (Xu & Knight, 2010; eq. 6),
BUT with weighting function `exp(-b^2 t^2 / 2)`, NOT `exp(-b t^2)`, as in the paper.

This is done purely to simplify formulae.
"""
struct DistanceXu{T<:Real} <: DistanceKind{T}
    """
    Strength of ECF smoothing.
    Higher `b` => stronger smoothing => much of the ECF is smoothed away
    """
    b::T

    function DistanceXu(b::T) where T<:Real
        @assert b > 0

        new{T}(b)
    end
end

"""
$TYPEDEF

Distance between mixture characteristic function (CF) and
the "Kernel Characteristic Function Estimator" (KCFE)
- CF of a kernel density estimator with window size `1/b`.

This distance is no longer weighted (weight = 1), and the hyperparameter `b`
doesn't affect the mixture CF anymore.
"""
struct DistanceKCFE{T<:Real} <: DistanceKind{T}
    "Inverse of KDE window size. Higher `b` => stronger smoothing."
    b::T

    function DistanceKCFE(b::T) where T<:Real
        @assert b > 0

        new{T}(b)
    end
end


"""
$TYPEDSIGNATURES

Part of the formula for the distance between mixture CF and ECF
that is constant w.r.t. mixture parameters.
See (Xu & Knight, 2010).

Here the weighting function is `exp(-b^2/2 * t^2)`, NOT `exp(-b t^2)` as in Xu.
"""
function distance_constant(::Type{DistanceXu{T}}, observations::AV{<:Real}, b::Real) where T<:Real
    sum(
        normal(observations[n], observations[m], b^2)
        for n in eachindex(observations), m in eachindex(observations)
    ) / (length(observations)^2)
end

"""
$TYPEDSIGNATURES

Part of the formula for the distance between mixture CF and KCFE
that is constant w.r.t. mixture parameters.
"""
distance_constant(::Type{DistanceKCFE{T}}, observations::AV{<:Real}, b::Real) where T<:Real =
    distance_constant(DistanceXu{T}, observations, sqrt(2) * b)

"""
$TYPEDSIGNATURES

Weighted distance between theoretical CF and the empirical CF.
See (Xu & Knight, 2010; eq. 6).
The weighting function used in this code is `exp(-b^2/2 * t^2)`, NOT `exp(-b t^2)` as in Xu.
"""
function distance(
    ::Type{DistanceXu{T}},
    p::AV{<:Real}, mu::AV{<:Real}, sigma::AV{<:Real}, observations::AV{<:Real}, b::Real,
    constant::Real=distance_constant(DistanceXu{T} ,observations, b)
) where T<:Real
    N = length(observations)

    part1 = -2/N  * sum(
        p[k] * normal(observations[n], mu[k], b^2 + sigma[k]^2)
        for n in eachindex(observations), k in eachindex(p)
    )
    part2 = sum(
        p[j] * p[k] * normal(mu[j], mu[k], b^2 + sigma[k]^2 + sigma[j]^2)
        for j in eachindex(p), k in eachindex(p)
    )

    part1 + part2 + constant
end

"""
$TYPEDSIGNATURES

Distance between theoretical characteristic function (CF)
and the Kernel CF Estimator (KCFE),
which is the CF of a simple kernel density estimator with window size `1/b > 0`.

- The theoretical CF is the model whose parameters we're estimating
- The KCFE is a dampened version of the empirical CF, where
the greater the window size `b`, the greater the dampening,
similar to density smoothing in regular KDE.

## Bias-variance tradeoff

- Greater `b` => greater dampening/smoothing of the empirical CF => greater bias.
- Lower `b` (maybe zero) => more rough empirical CF =>
more overfitting => greater variance.
"""
function distance(
    ::Type{DistanceKCFE{T}},
    p::AV{<:Real}, mu::AV{<:Real}, sigma::AV{<:Real}, observations::AV{<:Real}, b::Real,
    constant::Real=distance_constant(DistanceKCFE{T}, observations, b)
) where T<:Real
    N = length(observations)

    part1 = -2/N  * sum(
        p[k] * normal(observations[n], mu[k], b^2 + sigma[k]^2)
        for n in eachindex(observations), k in eachindex(p)
    )
    part2 = sum(
        # No `b^2` added to the variance!
        p[j] * p[k] * normal(mu[j], mu[k], sigma[k]^2 + sigma[j]^2)
        for j in eachindex(p), k in eachindex(p)
    )
    
    part1 + part2 + constant
end

"""
$TYPEDSIGNATURES

Same as `distance`, but all mixture parameters
come from one vector. Used for optimization.
"""
function distance_one_arg(
    kind::Type{<:DistanceKind},
    θ::AV{<:Real}, observations::AV{<:Real}, b::Real,
    constant::Real=distance_constant(kind, observations, b)
)
	p, mu, sigma = get_mix_params(θ)

	distance(kind, p, mu, sigma, observations, b, constant)
end
