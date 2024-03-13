push!(LOAD_PATH,"../src/")

using Documenter, UniversalDiffEq, DataFrames

makedocs(
    sitename="UniversalDiffEq.jl",
    modules  = [UniversalDiffEq],
    format   = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = ["index.md","Models.md","ModelTesting.md","NutsAndBolts.md","MultipleTimeSeries.md","NODEexample.md"]
)

deploydocs(
    repo = "github.com/jarroyoe/UniversalDiffEq.jl.git",
)