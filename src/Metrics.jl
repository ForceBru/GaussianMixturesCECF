module Metrics
export neg_log_likelihood, cauchy_schwarz, wasserstein

import ..AV, ..normal
using ComponentArrays, DocStringExtensions
import SpecialFunctions: erf

# ========== Log-likelihood ==========

"""
$TYPEDSIGNATURES

Log-likelihood of the mixture specified by `params`
given `data` used to fit the mixture.

__Higher is better__.
"""
function log_likelihood(params::ComponentVector, data::AV{<:Real})::Real
    p, mu, sigma = params.p, params.mu, params.sigma

    sum(
        sum(
            p[k] * normal(x, mu[k], sigma[k]^2)
            for k in eachindex(p)
        ) |> log
        for x in data
    )
end

"""
$TYPEDSIGNATURES

Negative log-likelihood of the mixture specified by `params`
given `data` used to fit the mixture.

__Lower is better__.
"""
neg_log_likelihood(params::ComponentVector, data::AV{<:Real})::Real =
    -log_likelihood(params, data)


# ========== Cauchy-Schwarz divergence ==========

function cauchy_schwarz_part(p::AV, mu::AV, tau::AV)
    s1 = sum(@. p^2 * sqrt(tau / (2π)) / sqrt(2))
    s2 = sum(
        sum(
            p[m] * p[m_] * normal(
                mu[m], mu[m_], 1/tau[m] + 1/tau[m_]
            )
            for m_ in 1:(m-1); init=0.0
        )
        for m in eachindex(p); init=0.0
    )

    0.5 * log(s1 + 2s2)
end

"""
$TYPEDSIGNATURES

Cauchy-Schwarz distance between the probability density function (PDF)
of the mixture specified by `params` and the kernel density estimate
of `data` with window size `h>0`.

__Lower is better__. Minimum: 0.

Paper: Kampa, Kittipat, Erion Hasanbelliu, and Jose C. Principe.
"Closed-Form Cauchy-Schwarz PDF Divergence for Mixture of Gaussians."
In The 2011 International Joint Conference on Neural Networks, 2578–85.
San Jose, CA, USA: IEEE, 2011. https://doi.org/10.1109/IJCNN.2011.6033555.
"""
function cauchy_schwarz(params::ComponentVector, data::AV{<:Real}, h::Real)::Real
    @assert h > 0
    N = length(data)
    p, mu, sigma = params.p, params.mu, params.sigma
    tau = @. 1 / sigma^2
    tau_data = 1 / h^2

    a = sum(
        p[m] * 1/N * normal(
            mu[m], x, 1/tau[m] + 1/tau_data
        )
        for m in eachindex(p), x in data
    ) |> log |> -

    (
        a
        + cauchy_schwarz_part(p, mu, tau) # fitted mixture
        + cauchy_schwarz_part(
            fill(1/N, N), data, fill(tau_data, N)
        ) # KDE of data (also mixture, but with common std `h`)
    )
end


# ========== 1-Wasserstein ==========

normal_CDF(x::Real) = (1 + erf(x / sqrt(2))) / 2
normal_CDF(x::Real, mu::Real, sigma::Real) = normal_CDF((x - mu) / sigma)
mix_CDF(x::Real, p::AV{<:Real}, mu::AV{<:Real}, sigma::AV{<:Real}) =
    sum(
        p_ * normal_CDF(x, mu_, sigma_)
        for (p_, mu_, sigma_) in zip(p, mu, sigma)
    )

struct EmpiricalCDF{T<:Real, D<:Integer}
    sorted_samples::Vector{T}
    CDF_values::Base.OneTo{D} # unnormalized!

    function EmpiricalCDF(samples::AV{T}) where T<:Real
        CDF_values = eachindex(samples)
        
        new{T, eltype(CDF_values)}(sort(samples), CDF_values)
    end
end

function (cdf::EmpiricalCDF{T})(x::Real) where T<:Real
    if x < cdf.sorted_samples[begin]
        return zero(T)
    elseif x > cdf.sorted_samples[end]
        return one(T)
    end

    GREATEST_CDF::T = T(cdf.CDF_values[end])

    for (i, sample) in enumerate(cdf.sorted_samples)
        (x < sample) && return cdf.CDF_values[i-1] / GREATEST_CDF
    end
end

# Results of `scipy.special.roots_legendre(9)`
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_legendre.html
const LEGENDRE_ROOTS = Float64[
    -0.9681602395076261, -0.8360311073266358, -0.6133714327005904,
    -0.3242534234038089,  0.                ,  0.3242534234038089,
     0.6133714327005904,  0.8360311073266358,  0.9681602395076261
]
const LEGENDRE_WEIGHTS = Float64[
    0.0812743883615741, 0.1806481606948576, 0.2606106964029355,
    0.3123470770400028, 0.3302393550012596, 0.3123470770400028,
    0.2606106964029355, 0.1806481606948576, 0.0812743883615741
]

# int_{-1}^1 f(t)dt
function integrate_gauss(f::Function)::Real
    return sum(
        w * f(t)
        for (w, t) in zip(LEGENDRE_WEIGHTS, LEGENDRE_ROOTS)
    )
end

integrate_gauss(f::Function, lo::Real, up::Real) =
    (up - lo)/2 * integrate_gauss(
        t->f( ((up - lo) * t + (up + lo)) / 2 )
    )

function integrate_gauss(f::Function, integration_range::StepRangeLen{<:AbstractFloat})
    the_range = integration_range
    return sum(
        integrate_gauss(f, lo, up)
        for (lo, up) in zip(the_range[begin:end-1], the_range[begin+1:end])
    )
end

function integrate_simpson(f::Function, integration_range::StepRangeLen{<:AbstractFloat})
    @assert iseven(length(integration_range))

    n = length(integration_range)
    xs = range(integration_range[begin], integration_range[end], n+1)
    Δx = xs[2] - xs[1]

    S1 = sum(f(x) for x in xs[2:2:n])
    S2 = sum(f(x) for x in xs[3:2:n-1])

    Δx/3 * (f(xs[begin]) + 4S1 + 2S2 + f(xs[end]))
end

function wasserstein(
    params::ComponentVector, F_data, integration_range::StepRangeLen;
    integrate::Function
)
    F_model(x::Real) = mix_CDF(x, params.p, params.mu, params.sigma)

    integrate(
        x->abs(F_data(x) - F_model(x)),
        integration_range
    )
end

"""
$TYPEDSIGNATURES

1-Wasserstein distance between the cumulative distribution function (CDF)
of the mixture specified by `params` and the empirical CDF of the `data`.

__Lower is better__. Minimum: 0.
"""
function wasserstein(
    params::ComponentVector, data::AV;
    nintervals::Integer=1000, mult::Real=10,
    integrate=integrate_gauss
)::Real
    @assert nintervals > 0
    @assert mult > 0
    min_, max_ = minimum(data), maximum(data)
    the_range = range(min_ - mult*abs(min_), max_ + mult*abs(max_), nintervals)

    wasserstein(params, EmpiricalCDF(data), the_range; integrate)
end

end # module
