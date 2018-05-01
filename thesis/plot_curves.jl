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
default = Pkg.dir("Multilane", "data/gap_Sunday_29_Apr_11_40.csv")
filenames = args["<filename>"]
if isempty(filenames)
    push!(filenames, default)
end
# @show filenames

data = vcat([CSV.read(f) for f in filenames]...)


data[:time_to_lane] = data[:steps_to_lane].*data[:dt]
data[:brakes_per_step] = data[:nb_brakes]./data[:n_steps]

combined = by(data, [:cor, :solver, :lambda]) do df
    n = size(df, 1)
    success = df[:terminal].=="lane"
    safe = (df[:nb_brakes] .< 1) .& (df[:min_speed] .>= 15.0)
    bpm = df[:nb_brakes]./df[:distance]
    return DataFrame(reached_lane=sum(success)/n,
                     rl_std=std(success),
                     safe=sum(safe)/n,
                     safe_std=std(safe),
                     nb_brakes=mean(df[:nb_brakes]),
                     nb_brakes_std=std(df[:nb_brakes]),
                     bpm=mean(bpm),
                     bpm_std=std(bpm),
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
        # plt = Plots.Linear(df[:safe], df[:reached_lane],
        plt = Plots.Linear(df[:bpm], df[:reached_lane],
                           legendentry=name,
                           errorBars=ErrorBars(x=df[:bpm_std]./sqrt.(df[:n]),
                                               y=df[:rl_std]./sqrt.(df[:n]))
                          )
        push!(plts, plt)
    end
    a = Axis(plts,
             title="Correlation $cor",
             ylabel="Fraction Successful",
             xlabel="Fraction Safe",
             # legendPos="south",
             # xmin=0.9,
             # xmax=1.0,
             ymin=0.0,
             ymax=1.0,
             style="legend style={at={(0.5,-0.2)}, anchor=north}"
            )

    file = tempname()*".pdf"
    save(file, a)
    run(`xdg-open $file`)
end
