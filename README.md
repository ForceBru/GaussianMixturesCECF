# GaussianMixturesCECF

Estimate Gaussian mixture models using the Continuous Empirical Characteristic Function (CECF) method introduced in (Xu & Knight, 2010).

The idea is to estimate parameters of a Gaussian mixture by minimizing the _weighted_ "distance" between the empirical characteristic function `CF_n(t)` and the theoretical one `CF(t; theta)`. The distance measure is given by the following integral (see eq. 6 from the paper):

![](img/distance.png)

As shown in the paper, for Gaussian mixtures this integral can be solved analytically,
which results in the following expressions for the distance measure:

![](img/eqns.png)

This is a simplified version of eq. 14 from the paper.

- `r` is the vector of data points (real numbers)
- `b > 0` is the parameter of the weighting function `exp(-b t^2)`, where `t in R` is the argument of the characteristic function.

The paper provides a method of calculating `b` automatically, but this is not yet implemented, so users should supply one explicitly.

## Usage

The API is Sklearn-like:

```julia
n_components = 3
gmm = GaussianMixture(n_components)
params_vector = fit_cecf!(gmm, data, b=0.01)
p, mu, sigma = get_mix_params(params_vector)
```

If multiple similar mixtures need to be estimated, `GaussianMixture` can keep track of the last estimates and use them as the initial guess for the optimizer. This can increase performance since optimization will likely begin near the optimum:

```julia
params_vector = fit_cecf!(gmm, data, b=0.01, update_guess=true)
```

One can also supply the initial guess to both `GaussianMixture` and `fit_cecf!`:

```julia
# Mixture of 2 components:
# p1, p2 = 0.5, 0.5
# mu1, mu2 = 0, 0
# sigma1, sigma2 = 1e-3, 2e-3
gmm = GaussianMixture([0.5, 0.5, 0, 0, 1e-3, 2e-3])
fit_cecf!(gmm, data, b=0.01, θ0=[0.5, 0.5, -1, 1, 1e-3, 2e-3])
```

## References

- Xu, Dinghai, and John Knight. 2010. "Continuous Empirical Characteristic Function Estimation of Mixtures of Normal Parameters." Econometric Reviews 30 (1): 25–50. <https://doi.org/10.1080/07474938.2011.520565>.
