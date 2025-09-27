using EpicHyperSketch
using Documenter

DocMeta.setdocmeta!(EpicHyperSketch, :DocTestSetup, :(using EpicHyperSketch); recursive=true)

makedocs(;
    modules=[EpicHyperSketch],
    authors="Shane Kuei-Hsien Chu (skchu@wustl.edu)",
    sitename="EpicHyperSketch.jl",
    format=Documenter.HTML(;
        canonical="https://kchu25.github.io/EpicHyperSketch.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kchu25/EpicHyperSketch.jl",
    devbranch="main",
)
