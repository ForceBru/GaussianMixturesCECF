"""
$TYPEDEF

State of the Gaussian mixture estimator.

$TYPEDFIELDS
"""
mutable struct GaussianMixture
    objective::Union{Optim.TwiceDifferentiable, Nothing}
    constraints::Optim.TwiceDifferentiableConstraints

    "Number of mixture components"
    K::Integer

    """
    Initial guess for optimizer.

    Format: `[weights; means; standard deviations;]`
    """
    θ0::AV{<:Real}
    "Parameter lower bounds (set automatically)"
    θ_lo::AV{<:Real}
    "Parameter upper bounds (set automatically)"
    θ_hi::AV{<:Real}

    "Result of optimization with Optim.jl"
    optim_result
end

"""
$TYPEDSIGNATURES

Initialize Gaussian mixture with initial guess `θ0`.
"""
function GaussianMixture(θ0::AV{<:Real})
    @assert length(θ0) % 3 == 0

    K = length(θ0) ÷ 3
    unbound_lo = fill(-Inf, K)
    unbound_hi = fill(Inf, K)

    # [weights; means; standard deviations]
    # Weights are between 0 and 1
    # Means are obviously unbounded
    # Standard deviations are unbounded TOO
    # because they'll be squared when computing distances.
    θ_lo = [zeros(K); unbound_lo; unbound_lo]
    θ_hi = [ ones(K); unbound_hi; unbound_hi]

    dfc = TwiceDifferentiableConstraints(
		constraint!,
        θ_lo, θ_hi, # parameter bounds
        [0.0], [0.0], # constraint bounds: equality constraint!
        :forward # use autodiff
	)
    GaussianMixture(
        nothing, dfc, # no objective yet
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
function constraint!(constr::AV{<:Real}, θ::AV{<:Real})
    p, _, _ = get_mix_params(θ)

    constr[1] = sum(p) - 1
    constr
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

- `b > 0` is the smoothing parameter
- `θ0` is the optional starting point
- `update_guess::Bool` - replace initial guess with newly found estimates?

Returns a `ComponentVector` with fields:

- `p` - mixture weights
- `mu` - mixture means
- `sigma` - mixture standard deviations
"""
function fit_cecf!(
    gm::GaussianMixture, data::AV{<:Real}; b::Real,
    θ0::AV{<:Real}=gm.θ0, update_guess::Bool=false
)::ComponentVector{<:Real}
    @assert b > 0
    @assert length(θ0) == 3gm.K
    _check_mix_params(θ0)

    gm.θ0 .= θ0

    # Will MINIMIZE this
    objective(θ::AV{<:Real}) = distance_one_arg(θ, data, b)
    gm.objective = TwiceDifferentiable(objective, θ0, autodiff=:forward)

    gm.optim_result = optimize(gm.objective, gm.constraints, gm.θ0, IPNewton())

    θ_est = Optim.minimizer(gm.optim_result)
    # Ensure non-negativity of standard deviations
    @. θ_est[2gm.K+1:end] = abs(θ_est[2gm.K+1:end])

    if update_guess
        gm.θ0 .= θ_est
    end

    ax = Axis(p=1:gm.K, mu=(gm.K+1):(2gm.K), sigma=(2gm.K+1):(3gm.K))
    ComponentVector(θ_est, ax)
end

"""
$TYPEDSIGNATURES

Convenience wrapper for `fit_cecf!` to quickly fit mixtures of `K` components
without creating the `GaussianMixture` object.
"""
fit_cecf(K::Integer, data::AV{<:Real}; b::Real) =
    fit_cecf!(GaussianMixture(K), data; b)
