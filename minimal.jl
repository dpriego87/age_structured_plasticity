module populations
export Individual, Population

import Base.copy!

type Individual
    genotype::Array{Float64,1}   # vector of genotypes

    # constructors
    Individual() = new(Float64[])
    Individual(genotype) = new(genotype)
end

function copy!(i::Individual, j::Individual)
    copy!(i.genotype, j.genotype)
end

function copy!(x::Array{Individual,1}, y::Array{Individual,1})
    if (length(x) != length(y))
        error("Arrays must have equal length")
    end
    
    for i=1:length(x)
        copy!(x[i], y[i])
    end
end


type Population
    size::Int64
    members1::Array{Individual,1}
    members2::Array{Individual,1}

    # create a population of individuals with one element vectors as genotypes
    function Population(size::Int64)
        
        members1 = Array{Individual}(size)
        for i=1:size
            members1[i] = Individual([0.0])
        end
        members2 = deepcopy(members1)

        new(size, members1, members2)
    end
end

end

### code test
using populations

# both members1 and members2 arrays are identical initially
p = Population(10)

# change the "genotype" vales for members1
for i=1:10
    p.members1[i].genotype = [1.0]
end

# members2 remains unaffected
println(p.members2)

# perform what should be a copy of the memory from members1 to members2
copy!(p.members2, p.members1)

# members2 should be updated
println(p.members2)

# change the "genotype" vales for members1
for i=1:10
    p.members1[i].genotype = [2.0]
end

# members2 has changed too, which shouldn't happen since only the memory was copied...?
println(p.members2)
