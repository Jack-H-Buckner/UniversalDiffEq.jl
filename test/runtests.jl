using UniversalDiffEq
using Test

@testset "UniversalDiffEq.jl" begin
    include("NODEtests.jl")
    include("BayesNODEtests.jl")
end
