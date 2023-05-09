using UniversalDiffEq
using Test

@testset "UniversalDiffEq.jl" begin
    include("test_NODE.jl")
    include("test_UDE.jl")
end
