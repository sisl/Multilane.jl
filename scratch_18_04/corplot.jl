using PGFPlots

using CSV
using DataFrames
using Missings
using Query

function only(a)
    @assert length(a) == 1
    return first(a)
end

filename = Pkg.dir("Multilane", "data/cor_trend_Monday_2_Apr_19_49.csv")

data = CSV.read(filename, nullable=true)

data[:time_to_lane] = data[:steps_to_lane].*data[:dt]
data[:brakes_per_step] = data[:nb_brakes]./data[:n_steps]

combined = by(data, [:cor, :solver]) do df
    n = size(df, 1)
    return DataFrame(reached_lane=sum(df[:terminal].=="lane")/n)
end

@show sort(combined, cols=[:solver])

plts = Plots.Plot[]
for df in groupby(combined, [:solver])
    df = sort(df, cols=[:cor])
    name = only(unique(df[:solver]))
    plt = Plots.Linear(convert(Vector{Float64}, df[:cor]), df[:reached_lane], legendentry=name)
    push!(plts, plt)
end
a = Axis(plts,
         ylabel="Fraction Successful",
         xlabel="Correlation",
         legendPos="south west"
        )

file = tempname()*".pdf"
save(file, a)
run(`xdg-open $file`)
