using PGFPlots

using CSV
using DataFrames
using Missings
using Query

function only(a)
    @assert length(a) == 1
    return first(a)
end

# filename = Pkg.dir("Multilane", "data/cor_trend_Monday_2_Apr_19_49.csv")
# filename = Pkg.dir("Multilane", "data/cor_trend_Tuesday_3_Apr_07_20.csv")
# filename = Pkg.dir("Multilane", "data/cor_trend_Tuesday_3_Apr_20_35.csv")
filename = Pkg.dir("Multilane", "data/cor_trend_Friday_27_Apr_03_39.csv")

data = CSV.read(filename)

data[:time_to_lane] = data[:steps_to_lane].*data[:dt]
data[:brakes_per_step] = data[:nb_brakes]./data[:n_steps]

data = @from i in data begin
    @where i.solver != "outcome"
    @select i
    @collect DataFrame
end

combined = by(data, [:cor, :solver]) do df
    return DataFrame(reached_lane=sum(df[:terminal].=="lane")/nrow(df),
                     n=nrow(df))
end

@show sort(combined, cols=[:solver])


plts = Plots.Plot[]
for df in groupby(combined, [:solver])
    df = sort(df, cols=[:cor])
    name = only(unique(df[:solver]))
    # 95% confidence region from Hoeffding Bound
    confidence = 0.5
    n = only(unique(df[:n]))
    confidence_radius = sqrt(log((1.0-confidence)/2)/(-2*n))
    plt = Plots.Linear(convert(Vector{Float64}, df[:cor]),
                       df[:reached_lane],
                       errorBars=ErrorBars(y=fill(confidence_radius, nrow(df))), 
                       legendentry=name)
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
