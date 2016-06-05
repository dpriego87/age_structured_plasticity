# Age-structured population simulation
#
# Jeremy Van Cleve <jeremy.vancleve@gmail.com>

module populations
export Individual, Population
export mean_genotype, mean_phenotype, next_gen

# import base functions for multiple dispatch
import Base.copy, Base.copy!
using Distributions

# base type for individuals in the population
type Individual
    age::Int64
    genotype::Array{Float64,1}   # vector of genotypes
    phenotype::Array{Float64,1}  # vector of phenotypes
    fitness::Array{Float64,1}    # (survival, fertility)

    # constructors
    Individual() = new(0, Float64[], Float64[], [1.0, 1.0])
    Individual(age) = new(age, Float64[], Float64[], [1.0, 1.0])
    Individual(age, genotype) = new(age, genotype, Float64[], [1.0, 1.0])
    Individual(age, genotype, phenotype) = new(age, genotype, phenotype, [1.0, 1.0])
    Individual(age, genotype, phenotype, fitness) = new(age, genotype, phenotype, fitness)
end

# copy method for Individual
function copy(i::Individual)
    j = Individual()
    j.age = i.age
    j.genotype = copy(i.genotype)
    j.phenotype = copy(i.phenotype)
    j.fitness = copy(i.fitness)
    return j
end

# copy inplace method for Individual
function copy!(i::Individual, j::Individual)
    i.age = j.age
    copy!(i.genotype, j.genotype)
    copy!(i.phenotype, j.phenotype)
    copy!(i.fitness, j.fitness)
end

# copy method for Array of Individuals
function copy(x::Array{Individual,1})
    y = Array{Individual}(size(x))
    for i=1:length(x)
        y[i] = copy(x[i])
    end
    return y
end

# copy inplace method for Array of Individuals
function copy!(x::Array{Individual,1}, y::Array{Individual,1})
    if (length(x) != length(y))
        error("Arrays must have equal length")
    end
    
    for i=1:length(x)
        copy!(x[i], y[i])
    end
end

# type for population, which is a collection of individuals
# and population-level properties
type Population
    size::Int64
    pheno_func::Function            # map genotype to phenotype
    fit_func::Function              # map phenotype to fitness
    mut_func::Function              # mutate genotype
    members::Array{Individual,1}
    members_prev::Array{Individual,1}
    fitness::Array{Float64,2}
    mean_fit::Array{Float64,1}
    env_func::Function
    env_state::Array{Float64,1}

    # Constructor
    ## geno_func function is used to initialize genotypes
    ## e.g., ()->[rand()] initializes each genotype to a random value in (0,1)
    function Population(size::Int64,
                        pheno_func::Function, fit_func::Function, mut_func::Function,
                        geno_func::Function, env_func::Function, env0::Array{Float64,1})
        
        members = Array{Individual}(size)
        for i=1:size
            g = geno_func(i)
            ind = Individual(0, g, Array(Float64,0))
            pheno_func(ind, env0)
            members[i] = ind
        end
        members_prev = copy(members)

        new(size, pheno_func, fit_func, mut_func, members, members_prev,
            hcat(fill(1, size), fill(1/size, size)), [1.0,1/size],
            env_func, env0)
    end
end

# update environmental state using previous state
function update_env_state(pop::Population)
    pop.env_state = pop.env_func(pop.env_state)
end

function calc_pheno_fitness(pop::Population)
    pop.mean_fit = [0.0, 0.0]

    for i = 1:pop.size
        pop.pheno_func(pop.members[i], pop.env_state)
        pop.fit_func(pop.members[i], pop.env_state)
        pop.fitness[i,:] = pop.members[i].fitness
        pop.mean_fit += pop.members[i].fitness
    end

    pop.mean_fit /= pop.size
end

function mean_genotype(pop::Population)
    ngeno = length(pop.members[1].genotype)
    pop_geno = Array(Float64, pop.size, ngeno)
    
    for i=1:pop.size
        pop_geno[i,:] = pop.members[i].genotype
    end

    return mean(pop_geno, 1)
end

function mean_phenotype(pop::Population)
    npheno = length(pop.members[1].phenotype)
    pop_pheno = Array(Float64, pop.size, npheno)
    
    for i=1:pop.size
        pop_pheno[i,:] = pop.members[i].phenotype
    end

    return mean(pop_pheno, 1)
end

function age_distribution(pop::Population)
    ages = Array(Float64, pop.size)

    for i=1:pop.size
        ages[i] = pop.members[i].age
    end
    return hist(ages)
end

###
### Main life cycle function:
### iterate through the lifecycle of the population once
### 1. fitness calculated
### 2. survival
### 3. fertility 
###
function next_gen(pop::Population)
    # save initial population state in "prev" vector
    copy!(pop.members_prev, pop.members)
    
    # update fitness values
    calc_pheno_fitness(pop)
    
    # get normalized fertility
    norm_fert = pop.fitness[:,2] / (pop.mean_fit[2] * pop.size)
    
    # set categorical distribution using normalized fertilities
    fertdist = Categorical(norm_fert)

    # survival and reproduction
    for i=1:pop.size
        if pop.members[i].fitness[1] > rand()
            # individual survives and ages
            pop.members[i].age += 1
        else
            # individual dies and is replaced by random new born 
            pop.members[i].age = 0
            parent = rand(fertdist)
            pop.members[i].genotype = pop.mut_func(pop.members_prev[parent])
        end
    end
    
    # update environmental state
    update_env_state(pop)
    
end

end
