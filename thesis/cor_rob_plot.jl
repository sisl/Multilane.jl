using PGFPlots

using CSV
using DataFrames
using Missings
using Query

function only(a)
    @assert length(a) == 1
    return first(a)
end

@show filename = Pkg.dir("Multilane", "data/cor_rob_Sunday_29_Jul_13_43.csv")

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

@show sort(combined, [:solver])

styles = Dict(
    "qmdp" => "blue",
    "pomcpow" => "brown",
    "meanmpc" => "black",
    "baseline" => "blue",
    "omniscient" => "red, dashed"
)
marks = Dict(
    "qmdp" => "*",
    "pomcpow" => "*",
    "meanmpc" => "star",
    "baseline" => "diamond*",
    "omniscient" => "*"
)

names = Dict("qmdp"=>"QMDP",
             "outcome"=>"Naive MDP",
             "pomcpow"=>"POMCPOW",
             "meanmpc"=>"Mean MPC",
             "baseline"=>"Assume normal",
             "omniscient"=>"Omniscient"
            )



plts = Plots.Plot[]
for df in groupby(combined, [:solver])
    df = sort(df, [:cor])
    name = only(unique(df[:solver]))
    # confidence region from Hoeffding Bound
    confidence = 0.68
    n = only(unique(df[:n]))
    confidence_radius = sqrt(log((1.0-confidence)/2)/(-2*n))
    plt = Plots.Linear(convert(Vector{Float64}, df[:cor]),
                       df[:reached_lane],
                       errorBars=ErrorBars(y=fill(confidence_radius, nrow(df))), 
                       # style="$(styles[name]), mar",
                       style="$(styles[name]), mark options={fill=$(styles[name])}",
                       legendentry=names[name],
                       mark=marks[name],
                      )
                      # legendentry=name)
    push!(plts, plt)
end
a = Axis(plts,
         ylabel="Fraction successful",
         xlabel="Correlation",
         # legendPos="south west"
         style="legend style={at={(0.5,-0.2)}, anchor=north}, legend columns=2, grid=both",
         width="3.5in",
         height="2.7in"
        )

# file = "/home/zach/Devel/thesis/media/corplot.tex"
# save(file, a, include_preamble=false)
file = "/home/zach/Devel/thesis/media/cor_rob.pdf"
save(file, a)
run(`xdg-open $file`)
