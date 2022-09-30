"""
$TYPEDEF

State of the Gaussian mixture estimator.

$TYPEDFIELDS
"""
mutable struct GaussianMixture{T<:Real}
    model::Union{ADNLPModel, Nothing}

    "Number of mixture components"
    K::Int

    """
    Initial guess for optimizer.

    Format: `[weights; means; standard deviations;]`
    """
    θ0::Vector{T}
    θ_lo::Vector{T}
    θ_hi::Vector{T}

    "Result of optimization"
    optim_result
end

"""
$TYPEDSIGNATURES

Initialize Gaussian mixture with initial guess `θ0`.
"""
function GaussianMixture(θ0::AV{T}) where T<:Real
    @assert length(θ0) % 3 == 0

    K = length(θ0) ÷ 3
    unbound_lo = fill(T(-Inf), K)
    unbound_hi = fill(T(Inf), K)

    # [weights; means; standard deviations]
    # Weights are between 0 and 1
    # Means are obviously unbounded
    # Standard deviations are unbounded TOO
    # because they'll be squared when computing distances.
    θ_lo = [zeros(T, K); unbound_lo; unbound_lo]
    θ_hi = [ ones(T, K); unbound_hi; unbound_hi]

    GaussianMixture{T}(
        nothing,
        K,
        θ0, θ_lo, θ_hi,
        nothing # no optimization result yet
    )
end

"""
$TYPEDSIGNATURES

Initialize Gaussian mixture with `n_components` components.
Initialize the initial guess automatically.
"""
function GaussianMixture(n_components::Integer; sigma_scale::Real=1e-3)
    @assert n_components > 1
    @assert sigma_scale > 0
    K = n_components

    θ0 = [ones(K) ./ K; zeros(K); (1:K) .* sigma_scale]
    GaussianMixture(θ0)
end

"""
$TYPEDSIGNATURES

Equality constraint: all weights must sum to one.
"""
function constraint(θ::AV{<:Real})
    p, _, _ = get_mix_params(θ)

    [sum(p) - 1]
end

function _check_mix_params(θ::AV)
    params = get_mix_params(θ)
    p = params.p

    @assert all(>=(0), p)
    @assert sum(p) ≈ 1
end

"""
$TYPEDSIGNATURES

Fit Gaussian mixture to `data`.

- `tol > 0` is the optimization tolerance
- `θ0` is the optional starting point
- `update_guess::Bool` - replace initial guess with newly found estimates?

Returns a `ComponentVector` with fields:

- `p` - mixture weights
- `mu` - mixture means
- `sigma` - mixture standard deviations
"""
function fit!(
    gm::GaussianMixture, data::AV{<:Real}, kind::DistanceKind;
    tol::Real=1e-6, use_log::Bool=false, eps::Real=1e-5,
    θ0::AV{<:Real}=gm.θ0, update_guess::Bool=false
)::ComponentVector{<:Real}
    @assert tol > 0
    @assert eps > 0
    @assert length(θ0) == 3gm.K
    _check_mix_params(θ0)

    kind_type = typeof(kind)
    b = kind.b
    gm.θ0 .= θ0

    constant = distance_constant(kind_type, data, b)
    # Will MINIMIZE this
    objective = if use_log
        θ->log(distance_one_arg(kind_type, θ, data, b, constant) + eps)
    else
        θ->distance_one_arg(kind_type, θ, data, b, constant)
    end

    model = ADNLPModel(
        objective,
        gm.θ0, gm.θ_lo, gm.θ_hi, # initial guess and bound constraints
        constraint, [0.0], [0.0] # equality constraint
    )

    gm.optim_result = Logging.with_logger(Logging.NullLogger()) do
        percival(model, ctol=tol)
    end

    θ_est = copy(gm.optim_result.solution)
    # Ensure non-negativity of standard deviations
    @. θ_est[2gm.K+1:end] = abs(θ_est[2gm.K+1:end])

    if update_guess
        gm.θ0 .= θ_est
    end

    ax = Axis(p=1:gm.K, mu=(gm.K+1):(2gm.K), sigma=(2gm.K+1):(3gm.K))
    ComponentVector(θ_est, ax)
end

fit!(gm::GaussianMixture, data::AV{<:Real}; b::Real, kwargs...) =
    fit!(gm, data, DistanceKCFE(b); kwargs...)

"""
$TYPEDSIGNATURES

Convenience wrapper for `fit_cecf!` to quickly fit mixtures of `K` components
without creating the `GaussianMixture` object.
"""
fit(K::Integer, data::AV{<:Real}, kind::DistanceKind; kwargs...) =
    fit!(GaussianMixture(K), data, kind; kwargs...)

fit(K::Integer, data::AV{<:Real}; b::Real, kwargs...) =
    fit(K, data, DistanceKCFE(b))
