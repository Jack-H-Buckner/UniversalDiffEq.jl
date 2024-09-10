using UniversalDiffEq
using Test

@testset "UniversalDiffEq.jl" begin 
    include("UDEtests.jl")
    include("NODEtests.jl")
    include("BayesNODEtests.jl")
    include("EasyNODEtests.jl")
end
