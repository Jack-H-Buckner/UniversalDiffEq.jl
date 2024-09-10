using UniversalDiffEq
using Test
print(pwd())
@testset begin #set "UniversalDiffEq.jl"
    include("UDEtests.jl")
    include("NODEtests.jl")
    include("BayesNODEtests.jl")
    include("EasyNODEtests.jl")
end
