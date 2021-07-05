using Documenter
using DocThemeIndigo
using Nabla

const indigo = DocThemeIndigo.install(Nabla)
makedocs(
    modules=[Nabla],
    format=Documenter.HTML(
        prettyurls=false,
        assets=[indigo],
    ),
    sitename="Nabla.jl",
    authors="Invenia Labs",
    pages=[
        "Home" => "index.md",
        "API" => "pages/api.md",
        "Custom Sensitivities" => "pages/custom.md",
        "Details" => "pages/autodiff.md",
    ],
)


deploydocs(
    repo = "github.com/invenia/Nabla.jl.git",
    push_preview=true,
)