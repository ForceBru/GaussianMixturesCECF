# Generate functions to compute gradients and Hessians using Symbolics.jl

The `D(r, b, θ)` function from Xu (eq. 7) can be represented in the standard "sum of functions `f` over data points" way, like for maximum likelihood.

![](../../img/sum_of_fns.png)

This lets us compute the observed Fisher matrix as the mean of `f`'s Hessians for different data poits `r_n`.

The functions in `gradient_*.jl` and `hessian_*.jl` is produced by Symbolics.jl run from `gen.jl`. To regenerate these files, run:

```
$ cd src/symbolics
$ julia --project gen_all.jl
```

Having a separate `Project.toml` in this directory seems ugly, but GaussianMixturesCECF doesn't depend on Symbolics at runtime, so I put it in a separate project.
