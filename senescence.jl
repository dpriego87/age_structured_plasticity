# Age-structured population simulation
#
# Jeremy Van Cleve <jeremy.vancleve@gmail.com>

using Distributions

include("fitness.jl")
include("environment.jl")

# base type for individuals in the population
type Individual
    age::Int16
    genotype::Array{Float64}   # vector of genotypes
    phenotype::Array{Float64}  # vector of phenotypes
    fitness::Array{Float64}    # (fertility, survival)

    # constructors
    Individual() = new(0, Float64[], [1.0, 1.0])
    Individual(age) = new(age, Float64[], Float64[], [1.0, 1.0])
    Individual(age, genotype) = new(age, genotype, Float64[], [1.0, 1.0])
    Individual(age, genotype, phenotype) = new(age, genotype, phenotype, [1.0, 1.0])
    Individual(age, genotype, phenotype, fitness) = new(age, genotype, phenotype, fitness)
end

# type for population, which is a collection of individuals
# and population-level properties
type Population
    size::Int64
    pheno_func::Function            # map genotype to phenotype
    fit_func::Function              # map phenotype to fitness
    mut_func::Function              # mutate genotype
    members::Array{Individual,1}
    fitness::Array{Float64,2}
    mean_fit::Array{Float64,1}
    env_func::Function
    env_state::Array{Float64,1}

    # Constructor
    ## geno function is used to initialize genotypes to arbitrary values
    ## e.g., ()->[rand()] initializes each genotype to a random value in (0,1)
    function Population(size, pheno_func, fit_func, mut_func,
                        geno::Function, env_func::Function, env0)
        
        members = Individual[]
        for i=1:size
            g = geno()
            push!(members, Individual(0, g, pheno_func(g, env0)))
        end

        new(size, pheno_func, fit_func, mut_func, members,
            hcat(fill(1, size), fill(1/size, size)), [1.0,1/size],
            env_func, env0)
    end
end

# update environmental state using previous state
function update_env_state(pop::Population)
    pop.env_state = pop.env_func(pop.env_state)
end

function calc_fitness(pop::Population)
    pop.mean_fit = [0.0, 0.0]

    for i = 1:pop.size
        pop.members[i].fitness = pop.fit_func(pop.members[i])
        pop.fitness[i,:] = pop.members[i].fitness
        pop.mean_fit += pop.members[i].fitness
    end

    pop.mean_fit /= pop.size
end

###
### Main life cycle function:
### iterate through the lifecycle of the population once
###
function next_gen(pop::Population)
    
    # update fitness values
    calc_fitness(pop)
    
    # get normalized fertility
    norm_fert = pop.fitness[:,1] / (pop.mean_fit[1] * pop.size)

    # set categorical distribution using normalized fertilities
    fertdist = Categorical(norm_fert)

    # survival and reproduction
    for i=1:pop.size
        if pop.members[i].fitness[2] > rand()
            # individual survives and ages
            pop.members[i].age += 1
        else
            # individual dies and is replaced by random new born 
            pop.members[i].age = 0
            parent = rand(fertdist)
            pop.members[i].genotype = pop.mut_func(pop.members[parent].genotype)
        end
    end

    # update environmental state
    update_env_state(pop)

end

# read command line arguments:
# population size, number of generations

if length(ARGS) != 2
    println("usage: simulation population_size num_generations")
    exit()
end

# convert arguments from strings to ints
(N, ngens) = map(x -> parse(Float64, x), ARGS)

pop = Population(N)

for i in 1:ngens
    next_gen(pop)
end

## TODO
##


for i in 1:10000
    next_gen(p)
    if i % 100 == 0
        println(p.mean_fit)
    end
end
