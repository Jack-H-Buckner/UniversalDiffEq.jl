using UniversalDiffEq
import Pkg; Pkg.add("Test")
using Test

@testset "UniversalDiffEq.jl" begin
    include("tests.jl")
end
