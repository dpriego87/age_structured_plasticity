## Plasticity and senescence simulation
##
## Jeremy Van Cleve <jeremy.vancleve@gmail.com>

include("populations.jl")
include("environment.jl")
include("phenotype.jl")
include("fitness.jl")

using Distributions
using ArgParse
using JLD
using populations

##
## Continuum of alleles, linear reaction norm with cost of slope.
## Test against theory from Lande (2014, JEB)
##

## Parameter object
type SenesceParams
    n::Int64
    s::Array{Float64,1}
    ve::Float64
    A::Float64
    B::Float64
    γ::Float64
    γb::Float64
    wmax::Float64
    venv::Float64
    arθ::Float64
    vmut::Float64
    reps::Int64
    burns::Int64
    iters::Int64

    # Constructor. Takes keyword arguments to make it easier to call with command line arguments
    SenesceParams(;n::Int64=100, s::Array{Float64,1}=[0.0], ve::Float64=0.01,
                  A::Float64=1.0, B::Float64=1.0, γ::Float64=2.0, γb::Float64=0.0, wmax::Float64=1.0,
                  venv::Float64=0.1, arθ::Float64=0.75, vmut::Float64=0.01,
                  reps::Int64=10, burns::Int64=5000, iters::Int64=1000) =
                      new(n, s, ve, A, B, γ, γb, wmax, venv, arθ, vmut, reps, burns, iters)
end


function runSim(p, params, verbose=true)
    reps, burns, iters = (params.reps, params.burns, params.iters)
    
    ngeno = length(p.members[1].genotype)
    nenv  = length(p.env_state)
    mgeno = zeros((ngeno, reps*iters))
    env   = zeros((nenv, reps*iters))

    totalruns = burns+iters
    iterblock = round(reps * totalruns / 100)
    
    for r = 1:reps
        # reset genotypes
        for i = 1:p.size
            p.members[i].genotype = zeros(ngeno)
        end

        for j = 1:totalruns
            next_gen(p)
            # record after burn in
            if j > burns
                mgeno[:,(r-1)*iters+j-burns] = mean_genotype(p)
                env[:,(r-1)*iters+j-burns]   = p.env_state
            end
            # print %done
            if ((r-1)*totalruns+j) % iterblock == 0 && verbose == true
                print(round(Int,100.*((r-1)*totalruns+j)/(totalruns*reps)), "% ")
            end
        end
    end
    println()

    return (mgeno, env)
end


## Assume a linear reaction norm with no past time dependence where
## eqn 2c in Lande (2014) becomes
## z_t = a + e + b ε_t
## Then the predicted evolved slope from eqn 6b is given below
function landeSlope(A, B, γ, γb, venv, arθ)
    # variance for autoregressive process of order one
    v = venv / (1 - arθ^2)
    # predicted slopes from Lande (2014, JEB)
    return [A, B / (1 + γb / (γ * v))]
end

function popLinearPlasticityFert(params)
    nages = length(params.s)
    stdenv = sqrt(params.venv)
    AB = [params.A, params.B]
    varθ = [params.arθ]

    function phenof(i::Individual, e::Array{Float64,1})
        copy!(i.phenotype, linear_norm_g(i.genotype[i.age*2+1:i.age*2+2], e, params.ve))
    end

    function fitf(i::Individual, e::Array{Float64,1})
        if i.age < nages
            i.fitness[1] = params.s[i.age+1]
        else
            i.fitness[1] = 0.0
        end
        i.fitness[2] = gauss_purify_cost_linear_norm(i.phenotype, e, AB, params.γ, params.γb, params.wmax)
    end

    function mutf(offspring::Individual, parent::Individual)
        copy!(offspring.genotype,
              parent.genotype + rand(Normal(0.0,params.vmut), size(parent.genotype)))
    end

    function envf(e::Array{Float64,1})
        env = zeros(2)
        env[1] = 1.0
        env[2] = autoregressive([e[2]], varθ, stdenv)[1]
        return env
    end

    p = Population(# population size
                   params.n,
                   # genotype->phenotype map
                   phenof, 3,
                   # fitness function
                   fitf,
                   # mutation function
                   mutf,
                   # initial genotype function
                   (i)->zeros(2*nages),
                   envf, # env update function
                   [1.0, 0.0]); # initial env state

    return p
end

function popLinearPlasticitySurv(params)
    nages = length(params.s)
    stdenv = sqrt(params.venv)
    AB = [params.A, params.B]
    varθ = [params.arθ]
    
    function phenof(i::Individual, e::Array{Float64,1})
        copy!(i.phenotype, linear_norm_g(i.genotype[i.age*2+1:i.age*2+2], e, params.ve))
    end

    function fitf(i::Individual, e)
        if i.age < nages - 1
            i.fitness[1] = params.s[i.age+1] +
            gauss_purify_cost_linear_norm(i.phenotype, e, AB, params.γ, params.γb, params.wmax)
        else
            i.fitness[1] = 0.0
        end
        i.fitness[2] = 1.0
    end

    function mutf(offspring::Individual, parent::Individual)
        copy!(offspring.genotype,
              parent.genotype + rand(Normal(0.0,params.vmut), size(parent.genotype)))
    end

    function envf(e::Array{Float64,1})
        env = zeros(2)
        env[1] = 1.0
        env[2] = autoregressive([e[2]], varθ, stdenv)[1]
        return env
    end

    p = Population(# population size
                   params.n,
                   # genotype->phenotype map
                   phenof, 3,
                   # fitness function
                   fitf,
                   # mutation function
                   mutf,
                   # initial genotype function
                   (i)->zeros(2*nages),
                   envf, # env update function
                   [1.0, 0.0]); # initial env state

    return p
end

## Tell ArgParse how to read Arrays from string arguments 
function ArgParse.parse_item(::Type{Array{Float64,1}}, x::AbstractString)
    return Array{Float64,1}(eval(parse(x)))
end

function main()

    s = ArgParseSettings()

    pars = ["n", "s", "ve", "A", "B", "γ", "γb",
            "wmax", "venv", "arθ", "vmut",
            "reps", "burns", "iters", "file"]
    parstype = [Int64, Array{Float64,1}, Float64, Float64, Float64, Float64, Float64,
                Float64, Float64, Float64, Float64,
                Int64, Int64, Int64, String]
    
    args = Array{Any,1}()
    for i in 1:length(pars)
        push!(args, "--"*pars[i])
        push!(args, Dict(:arg_type => parstype[i], :required => true))
    end
    add_arg_table(s, args...)
    parsed_args = parse_args(s)
    sim_args = copy(parsed_args)
    delete!(sim_args, "file")

    sim_params = Dict()
    for (arg, val) in sim_args
        sim_params[parse(arg)] = val
    end

    # create parameter object, pop, and run sim
    params = SenesceParams(;sim_params...)
    # println(params)
    # exit()
    pop = popLinearPlasticityFert(params)
    (mg, env) = runSim(pop, params)

    # save results (appending .jl if necessary)
    file = ismatch(r"\.jl", parsed_args["file"]) ? parsed_args["file"] : parsed_args["file"]*".jl"
    save(file, "params", params, "mg", mg, "env", env)

end

main()
