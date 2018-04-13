using PGFPlots

using CSV
using DataFrames
using Missings
using Query

function only(a)
    @assert length(a) == 1
    return first(a)
end

# filename = Pkg.dir("Multilane", "data/baseline_curve_Sunday_1_Apr_02_18.csv")
# filename = Pkg.dir("Multilane", "data/baseline_curve_Monday_2_Apr_14_49.csv")
# filename = Pkg.dir("Multilane", "data/baseline_curve_Monday_2_Apr_14_49.csv")
# filename = Pkg.dir("Multilane", "data/all_gaps_Monday_2_Apr_21_34.csv")
# filename = Pkg.dir("Multilane", "data/all_gaps_Wednesday_4_Apr_04_20.csv") # New filter
# filename = Pkg.dir("Multilane", "data/all_gaps_Thursday_5_Apr_00_53.csv") # Old Filter, N=2000
# filename = Pkg.dir("Multilane", "data/all_gaps_Monday_2_Apr_21_34.csv")
# filename = Pkg.dir("Multilane", "data/all_gaps_Wednesday_4_Apr_04_20.csv")
filename = Pkg.dir("Multilane", "data/all_gaps_Friday_13_Apr_10_17.csv")

data = CSV.read(filename, nullable=true)

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
