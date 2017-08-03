using Documenter, Nabla

makedocs(
    modules=[Nabla],
    format=:html,
    pages=[
        "Home" => "index.md",
        "API" => "pages/api.md"
    ],
    sitename="Nabla.jl",
    authors="Invenia Labs",
    assets=[
        "assets/invenia.css",
    ],
)

#=deploydocs(
    repo = "github.com/invenia/Nabla.jl.git",
    julia = "0.6",
    target = "build",
    deps = nothing,
    make = nothing,
)=#
