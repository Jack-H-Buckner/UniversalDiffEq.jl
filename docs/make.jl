push!(LOAD_PATH,"../src/")
using Pkg; Pkg.add("DataFrames")
using Documenter, UniversalDiffEq, DataFrames

makedocs(
    sitename="UniversalDiffEq.jl",
    modules  = [UniversalDiffEq],
    format   = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = ["index.md","Models.md","ModelTesting.md","NutsAndBolts.md","MultipleTimeSeries.md","modelanalysis.md","examples.jl"]
)

deploydocs(
    repo = "github.com/Jack-H-Buckner/UniversalDiffEq.jl.git",
)