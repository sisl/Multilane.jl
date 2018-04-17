using PGFPlots

using CSV
using DataFrames
using Missings
using Query
using DocOpt

doc = """

Usage:

"""

args = docopt(doc)

function only(a)
    @assert length(a) == 1
    return first(a)
end

# filename = Pkg.dir("Multilane", "data/all_gaps_Friday_13_Apr_10_17.csv")
filename = Pkg.dir("Multilane", "data/all_gaps_Monday_16_Apr_15_05.csv")

data = CSV.read(filename)

data[:time_to_lane] = data[:steps_to_lane].*data[:dt]
data[:brakes_per_step] = data[:nb_brakes]./data[:n_steps]

combined = by(data, [:cor, :solver, :lambda]) do df
    n = size(df, 1)
    return DataFrame(reached_lane=sum(df[:terminal].=="lane")/n,
                     had_brakes=sum(df[:nb_brakes].>=1)/n,
                    )
end

for cordf in groupby(combined, [:cor])
    @show sort(cordf, cols=[:solver])
    cor = only(unique(cordf[:cor]))
    plts = Plots.Plot[]
    for df in groupby(cordf, [:solver])
        df = sort(df, cols=[:lambda])
        name = only(unique(df[:solver]))
        plt = Plots.Linear(1.0.-df[:had_brakes], df[:reached_lane], legendentry=name)
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
