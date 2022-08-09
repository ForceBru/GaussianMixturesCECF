"""
$(README)

$(EXPORTS)
"""
module GaussianMixturesCECF
export GaussianMixture, fit_cecf!, fit!, fit_cecf, fit, get_mix_params

using DocStringExtensions
using Optim, ComponentArrays

const AV = AbstractVector{T} where T

include("Misc.jl")
include("Distance.jl")
include("Fit.jl")

end # module
