using UniversalDiffEq
using Test

@testset begin #set "UniversalDiffEq.jl"
    include("UDEtests.jl")
    include("NODEtests.jl")
    include("BayesNODEtests.jl")
    include("EasyNODEtests.jl")
end
