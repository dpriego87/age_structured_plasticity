###
### Run plasticity senescence simulation in batch using SLURM
###

using PyPlot
using JLD

pars = ["n", "s", "ve", "A", "B", "γ", "γb",
        "wmax", "venv", "arθ", "vmut",
        "reps", "burns", "iters"]

parvals = [[500,250], [[0.9, 0.0]], [0.01], [1], [2], [1], [0],
           [1], [0.1], [0.75], [0.01],
           [50], [10000], [1000]]

basename = "data"

## cartesian product of parameter values (all combinations)
parsets = collect(Base.product(parvals...))
nsets = length(parsets)

for p in 1:nsets
    simstr = "senescence.jl "
    
    # create filname base from basename and run number
    numstr = lpad(string(p), length(string(nsets)), "0")
    filebase = basename * numstr

    # add parameter values as command line options
    simstr *= join( map( (x) -> "--" * x[1] * "=\"" * string(x[2]) * "\"", zip(pars, parsets[p])), " ")
    simstr *= " --file=" * filebase * ".jl"

    cmdstr = "sbatch --job-name='plasticity_aging_" * numstr *
        "' --output=" * basename * ".out --wrap='" * simstr * "'"

    println(cmdstr)
end
