push!(LOAD_PATH,"../src/")
import Pkg; Pkg.add("DataFrames")
using Documenter, UniversalDiffEq, DataFrames

makedocs(
    sitename="UniversalDiffEq.jl",
    modules  = [UniversalDiffEq],
    format   = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = ["index.md","Models.md","ModelTesting.md","NutsAndBolts.md","MultipleTimeSeries.md"]
)

deploydocs(
    repo = "github.com/jarroyoe/UniversalDiffEq.jl.git",
)