using UniversalDiffEq
using Test

@testset "UniversalDiffEq.jl" begin 
    include("UDETests.jl")
    include("NODEtests.jl")
    include("BayesNODEtests.jl")
    include("EasyNODEtests.jl")
end
