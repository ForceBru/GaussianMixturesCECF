# Generate gradients and Hessians of `f` (see `gen.jl`)
# for some values of `K` (number of components)
@info "Generating gradients and Hessians..."
include("gen.jl")

# Only up to 4 components, because for 5
# the code for Hessian is literally 1MB
for what in [:gradient, :hessian], K in 2:4
    fname = "$(what)_$K.jl"

    @info "Computing" what K
    local code, sz = fn_source(what, K, true)

    @info "Writing" fname
    write(fname, code)
end

@info "Done!"
