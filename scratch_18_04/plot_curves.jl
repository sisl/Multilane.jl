using PGFPlots

using CSV
using DataFrames
using Missings
using DocOpt

doc = """
Usage:
    plot_curves.jl [<filename>...]
"""

args = docopt(doc)

function only(a)
    @assert length(a) == 1
    return first(a)
end

# default = Pkg.dir("Multilane", "data/all_gaps_Friday_13_Apr_10_17.csv")
default = Pkg.dir("Multilane", "data/all_gaps_Monday_16_Apr_15_05.csv")
filenames = args["<filename>"]
if isempty(filenames)
    push!(filenames, default)
end
@show filenames

data = vcat([CSV.read(f) for f in filenames]...)


data[:time_to_lane] = data[:steps_to_lane].*data[:dt]
data[:brakes_per_step] = data[:nb_brakes]./data[:n_steps]

combined = by(data, [:cor, :solver, :lambda]) do df
    n = size(df, 1)
    return DataFrame(reached_lane=sum(df[:terminal].=="lane")/n,
                     rl_std=std(df[:terminal].=="lane"),
                     had_brakes=sum(df[:nb_brakes].>=1)/n,
                     hb_std=std(df[:nb_brakes].>=1),
                     n=n
                    )
end

for cordf in groupby(combined, [:cor])
    @show sort(cordf, cols=[:solver])
    cor = only(unique(cordf[:cor]))
    plts = Plots.Plot[]
    for df in groupby(cordf, [:solver])
        df = sort(df, cols=[:lambda])
        name = only(unique(df[:solver]))
        plt = Plots.Linear(1.0.-df[:had_brakes], df[:reached_lane],
                           legendentry=name,
                           errorBars=ErrorBars(x=df[:hb_std]./sqrt.(df[:n]),
                                               y=df[:rl_std]./sqrt.(df[:n]))
                          )
        push!(plts, plt)
    end
    a = Axis(plts,
             title="Correlation $cor",
             ylabel="Fraction Successful",
             xlabel="Fraction Safe",
             legendPos="south west"
            )

    file = tempname()*".pdf"
    save(file, a)
    run(`xdg-open $file`)
end
