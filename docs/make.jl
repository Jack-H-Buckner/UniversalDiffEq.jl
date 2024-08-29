push!(LOAD_PATH,"../src/")
using Pkg; Pkg.add("DataFrames")
using Documenter, UniversalDiffEq, DataFrames

makedocs(
    sitename="UniversalDiffEq.jl",
    modules  = [UniversalDiffEq],
    format   = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = ["index.md","Models.md","EasyModels.md","ModelTesting.md","NutsAndBolts.md","MultipleTimeSeries.md","modelanalysis.md","BayesianModels.md","examples.md","API.md"]
)

deploydocs(
    repo = "github.com/Jack-H-Buckner/UniversalDiffEq.jl.git",
)