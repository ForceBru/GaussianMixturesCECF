using Symbolics

const AV = AbstractVector{T} where T

# Use `2.0` everywhere, because with `2`
# Symbolics generates `2//1` (!),
# but we need a simple `Float64` here
normal_pdf(x::Real, mu::Real, var::Real, π::Real) =
    exp(-(x - mu)^2 / (2.0var)) / sqrt(2.0π * var)

"""
Function such that Xu's distance `D` is:

    D(rs, b, αs, μs, σs^2) = mean(
        f(r, b, αs, μs, σs^2)
        for r in rs
    )
"""
f(r::Real, b::Real, π::Real, ws::AV, mus::AV, stds::AV)::Real =
    sum(
        2.0normal_pdf(r, mu_k, 2.0b + var_k, π)
        - w_k * sum(
            w_h * normal_pdf(mu_k, mu_h, 2.0b + var_k + var_h, π)
            for (w_h, mu_h, var_h) in zip(ws, mus, stds)
        )
        for (w_k, mu_k, var_k) in zip(ws, mus, stds)
    )

function compute_gradient(K::Integer, simplify::Bool)
    @assert K > 0
    @variables r b π ws[1:K] mus[1:K] stds[1:K]

    (;
        args=[r, b, π, ws, mus, stds],
        expr=Symbolics.gradient(f(r, b, π, ws, mus, collect(stds .^2)), [ws; mus; stds]; simplify),
        storage_size=(3K, )
    )
end

"""
Hessian of the _single_ function `f`.
"""
function compute_Hessian(K::Integer, simplify::Bool)
    @assert K > 0
    @variables r b π ws[1:K] mus[1:K] stds[1:K]

    (;
        args=[r, b, π, ws, mus, stds],
        expr=Symbolics.hessian(f(r, b, π, ws, mus, collect(stds .^2)), [ws; mus; stds]; simplify),
        storage_size=(3K, 3K)
    )
end

const WHATS = (:hessian, :gradient)

"""
Generate source code of a function that writes
the `what` (gradient/Hessian) of `f` (see above)
to the given array.

- `K` - number of mixture components
- `mut` - whether the function should be mutating
- `simplify` should probably remain `false`.
Otherwise Symbolics generates huge source code...
"""
function fn_source(what::Symbol, K::Integer, mut::Bool; simplify::Bool=false)::Tuple{String, Tuple}
    res = Dict(
        :hessian=>compute_Hessian, :gradient=>compute_gradient
    )[what](K, simplify)

    fn, fn_mut = build_function(res.expr, res.args...)

    the_fn = mut ? fn_mut : fn
    the_fn = Base.remove_linenums!(the_fn)

    fn_code = """
    \"\"\"
    $what of a single `f` function (see `gen.jl`) for $K mixture components.

    INPUT
    ==========
    - $(res.args[1])::Real (input data point)
    - $(res.args[2])::Real
    - $(res.args[3])::Real (must be literal pi=3.1415926...)
    - $(res.args[4])::Vector{Real} (mixture weights)
    - $(res.args[5])::Vector{Real} (mixture means)
    - $(res.args[6])::Vector{Real} (mixture standard deviations)
    
    OUTPUT
    ==========
    - Type: Vector{Real}
    - Shape: $(res.storage_size)
    \"\"\"
    $(what)_$(K)! = $the_fn
    """

    fn_code, res.storage_size
end
