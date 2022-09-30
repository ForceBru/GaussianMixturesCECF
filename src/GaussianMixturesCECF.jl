"""
$(README)

$(EXPORTS)
"""
module GaussianMixturesCECF
export GaussianMixture, fit_cecf!, fit!, fit_cecf, fit

import Logging
using DocStringExtensions
using ComponentArrays

import ADNLPModels: ADNLPModel
import Percival: percival

const AV = AbstractVector{T} where T

include("Misc.jl")
include("Distance.jl")
include("Fit.jl")
include("Metrics.jl")

end # module
