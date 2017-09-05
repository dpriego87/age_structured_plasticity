include("senescence.jl")

using JLD
using ArgParse

## Tell ArgParse how to read Arrays from string arguments 
function ArgParse.parse_item(::Type{Array{Float64,1}}, x::AbstractString)
    return Array{Float64,1}(eval(parse(x)))
end
function ArgParse.parse_item(::Type{Array{Bool,1}}, x::AbstractString)
    return Array{Bool,1}(eval(parse(x)))
end

function main()

    s = ArgParseSettings()

    # gather parameters from SenesceParams object automatically and add file option
    pars = [["--"*String(field) for field in fieldnames(SenesceParams)];
            ["--file"]]
    parstype = [[fieldtype(SenesceParams,field) for field in fieldnames(SenesceParams)];
                String]
    
    args = Array{Any,1}()
    for i in 1:length(pars)
        push!(args, pars[i])
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
    pop = popLinearPlasticity(params)
    (mg, env) = runSim(pop, params)

    # save results (appending .jld if necessary)
    file = ismatch(r"\.jld", parsed_args["file"]) ? parsed_args["file"] : parsed_args["file"]*".jld"
    save(file, "params", params, "mg", mg, "env", env)

end

main()
