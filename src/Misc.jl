"Normal PDF"
@inline normal(x::Real, mu::Real, var::Real) =
    exp(-(x - mu)^2 / (2var)) / sqrt(2π * var)
