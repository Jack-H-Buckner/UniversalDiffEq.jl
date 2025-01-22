using UniversalDiffEq
using Test

@testset "UniversalDiffEq.jl" begin 
    include("UDETests.jl")
    include("NODE.jl")
    include("MultiUDE.jl")
    # include("BayesNODEtests.jl")
    # include("EasyNODEtests.jl")
end
