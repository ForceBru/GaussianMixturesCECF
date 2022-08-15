module ExpGrad

import ..get_mix_params, ..distance_one_obs

import ForwardDiff, Tullio

const AV = AbstractVector{T} where T

normal_pdf(x::Real, mu::Real, var::Real) = exp(-(x - mu)^2 / (2var)) / sqrt(2π * var)

function distance(observations::AV{<:Real}, p::AV{<:Real}, mu::AV{<:Real}, sigma::AV{<:Real}, b::Real)
    # Tullio.@tullio penalty[1] := p[k] * p[h] * normal_pdf(mu[k], mu[h], 2b + sigma[k]^2 + sigma[h]^2) grad=Dual
    # Tullio.@tullio tmp[1] := p[k] * normal_pdf(observations[n], mu[k], 2b + sigma[k]^2) grad=Dual

    # -2/length(observations) * sum(tmp) + sum(penalty)

    idx = eachindex(p)
    
    penalty = sum(
        p[k] * p[h] * normal_pdf(mu[k], mu[h], 2b + sigma[k]^2 + sigma[h]^2)
        for k ∈ idx, h ∈ idx
    )

    -2/length(observations) * sum(
        distance_one_obs(p, mu, sigma, r, b) for r ∈ observations
    ) + penalty
end

@inline distance(observations::AV{<:Real}, θ::AV{<:Real}, b::Real) =
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

function GaussianMixture(K::Integer)
    w0 = ones(K) ./ K

    m0 = zeros(K)
    s0 = (1:K) .* rand()

    GaussianMixture(K, [w0; m0; s0])
end

abstract type AbstractOptimizer end

mutable struct ADAM{T<:Real, U<:Real} <: AbstractOptimizer
    t::UInt
    β1::T
    β2::T
    eps::T

    grad::Vector{U}
    m::Vector{U}
    v::Vector{U}
    m_hat::Vector{U}
    v_hat::Vector{U}
end

function ADAM(β1::T=0.9, β2::T=0.999, eps::T=1e-8) where T<:Real
    ADAM(
        UInt64(0x0), # time
        β1, β2, eps,
        # Dummy values
        zeros(T, 1),
        zeros(T, 1), zeros(T, 1),
        zeros(T, 1), zeros(T, 1),
    )
end

function update_grad!(opt::ADAM{T, U}, grad::AV{<:Real}) where {T, U<:Real}
    if opt.t == 0x0
        opt.grad = zeros(U, size(grad)...)
        opt.m = copy(opt.grad)
        opt.v = copy(opt.m)
        opt.m_hat = copy(opt.m)
        opt.v_hat = copy(opt.v)
    end

    opt.t += 0x1
    @. opt.m = opt.β1 * opt.m + (1 - opt.β1) * grad
    @. opt.v = opt.β2 * opt.v + (1 - opt.β2) * grad^2
    @. opt.m_hat = opt.m / (1 - opt.β1^opt.t)
    @. opt.v_hat = opt.v / (1 - opt.β2^opt.t)

    @. opt.grad = opt.m_hat / (sqrt(opt.v_hat) + opt.eps)
    nothing
end

function project_on_probability_simplex(y::AV{<:Real})
    u = sort(y, rev=true) # u1 ≥ u2 ≥ u3 ...
    
    idx = eachindex(u)
    sums = (1 .- cumsum(u)) ./ idx

    rho = maximum(
        j
        for j in eachindex(u)
        if u[j] + sums[j] > 0
    )
    λ = sums[rho]

    max.(y .+ λ, 0)
end

function fit_cecf!(
    mix::GaussianMixture, sample::AV{<:Real}; b::Real,
    opt::AbstractOptimizer=ADAM(),
    constr=:exp,
    lr::Real=1e-3, tol::Real=1e-6, quiet::Bool=true
)
    @assert b ≥ 0
    @assert lr > 0
    @assert tol > 0

    K = mix.K

    objective = θ -> distance(sample, θ, b)
    cfg_grad = ForwardDiff.GradientConfig(objective, mix.θ0)

    θ = copy(mix.θ0)
    θ_lag = zero(θ)

    grad = similar(θ)

    itr::Int64 = 0
    metric_ok = 0
    obj_lag = Inf
    objectives = fill(Inf, 30)
    while true
        itr += 1
        θ_lag .= θ

        # 1. Compute gradient
        ForwardDiff.gradient!(grad, objective, θ, cfg_grad)

        # 2. Smooth gradient
        update_grad!(opt, grad)

        @views begin
            if constr == :exp
                # 3. Exponentiated grad descent w.r.t. weights
                @. θ[1:K] = θ[1:K] * exp(-lr * opt.grad[1:K])
                θ[1:K] ./= sum(θ[1:K])
            elseif constr == :proj
                # 3. Projected gradient descent
                @. θ[1:K] -= lr * opt.grad[1:K]
                θ[1:K] .= project_on_probability_simplex(θ[1:K])
            else
                error("fuck")
            end

            # 4. Gradient descent step w.r.t. everything else
            @. θ[K+1:end] -= lr * opt.grad[K+1:end]
        end

        objectives[1:end-1] .= @view objectives[2:end]
        objectives[end] = objective(θ)

        if all(isfinite, objectives)
            metric = @views sum(
                abs(x - y)
                for (x, y) in zip(objectives[1:end-1], objectives[2:end])
            )
        else
            metric = Inf
        end

        (metric < tol) && (metric_ok += 1)
        (metric_ok == 50) && break
        
        if !quiet && itr % 10_000 == 0
            @show itr, metric, objectives[end]
        end
    end

    @show objective(θ)

    θ
end

fit

end