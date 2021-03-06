"""
$(README)

$(EXPORTS)
"""
module GaussianMixturesCECF
export GaussianMixture, fit_cecf!, fit_cecf, get_mix_params

using DocStringExtensions
using Optim

const AV = AbstractVector{T} where T

include("Misc.jl")
include("Distance.jl")
include("Fit.jl")

end # module
