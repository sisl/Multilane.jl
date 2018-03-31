using PGFPlots

using CSV
using DataFrames
using Missings
using Query

function only(a)
    @assert length(a) == 1
    return first(a)
end

# filename = Pkg.dir("Multilane", "data/baseline_curve_Saturday_24_Mar_21_08.csv")
# filename = Pkg.dir("Multilane", "data/baseline_curve_Sunday_25_Mar_12_09.csv")
# filename = Pkg.dir("Multilane", "data/baseline_curve_Wednesday_28_Mar_16_55.csv")
filename = Pkg.dir("Multilane", "data/baseline_curve_Friday_30_Mar_07_36.csv")

data = CSV.read(filename, nullable=true)

data[:time_to_lane] = data[:steps_to_lane].*data[:dt]
data[:brakes_per_step] = data[:nb_brakes]./data[:n_steps]

combined = by(data, [:solver, :lambda]) do df
    n = size(df, 1)
    return DataFrame(reached_lane=sum(df[:terminal].=="lane")/n,
                     had_brakes=sum(df[:nb_brakes].>=1)/n,
                    )
end

@show sort(combined, cols=[:solver])

plts = Plots.Plot[]
for df in groupby(combined, [:solver])
    df = sort(df, cols=[:lambda])
    name = only(unique(df[:solver]))
    plt = Plots.Linear(1.0.-df[:had_brakes], df[:reached_lane], legendentry=name)
    push!(plts, plt)
end
a = Axis(plts,
         ylabel="Fraction Successful",
         xlabel="Fraction Safe",
         legendPos="south west"
        )

file = tempname()*".pdf"
save(file, a)
run(`xdg-open $file`)
