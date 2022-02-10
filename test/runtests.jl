using Test

import Optim
using Distributions
using GaussianMixturesCECF

function sample_mixture(params::NamedTuple, N::Integer)
    distr = UnivariateGMM(params.mu, params.sigma, Categorical(params.p))

    rand(distr, N)
end

"Sort mixture params w.r.t. weights `p`"
function sort_params(params::NamedTuple)
    idx = sortperm(params.p)

    (; p=params.p[idx], mu=params.mu[idx], sigma=params.sigma[idx])
end

@testset "Simple 2 components" begin
    correct = (p=[.4, .6], mu=[-1, 1], sigma=[.2, .3]) |> sort_params
    data = sample_mixture(correct, 1000)

    @testset "Init with initial guess" begin
        gmm = GaussianMixture([0.5, 0.5, 0,0, 1e-3, 2e-3])
        estimated = fit_cecf!(gmm, data, b=0.01) |> get_mix_params |> sort_params

        display(gmm.optim_result)
        @test Optim.converged(gmm.optim_result)

        @info "Estimation results" correct estimated

        @test estimated.p ≈ correct.p atol=0.1
        @test estimated.mu ≈ correct.mu atol=0.1
        @test estimated.sigma ≈ correct.sigma atol=0.1
    end

    @testset "Init with number of components" begin
        gmm = GaussianMixture(2) # 2 components
        estimated = fit_cecf!(gmm, data, b=0.01) |> get_mix_params |> sort_params

        @test Optim.converged(gmm.optim_result)

        @test estimated.p ≈ correct.p atol=0.1
        @test estimated.mu ≈ correct.mu atol=0.1
        @test estimated.sigma ≈ correct.sigma atol=0.1
    end
end

@testset "Simple 3 components" begin
    correct = (p=[.4, .35, .25], mu=[-1, 0, 1], sigma=[.2, 0.1, .3]) |> sort_params
    data = sample_mixture(correct, 1000)

    gmm = GaussianMixture([ones(3) ./ 3; zeros(3); (1:3) .* 1e-3])
    estimated = fit_cecf!(gmm, data, b=0.01) |> get_mix_params |> sort_params

    @test Optim.converged(gmm.optim_result)

    @test estimated.p ≈ correct.p atol=0.1
    @test estimated.mu ≈ correct.mu atol=0.1
    @test estimated.sigma ≈ correct.sigma atol=0.1
end
