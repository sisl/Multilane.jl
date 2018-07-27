using PGFPlots

using CSV
using DataFrames
using Missings
using DocOpt
using Multilane

# pushPGFPlotsOptions("scale=1.2")

doc = """
Usage:
    plot_curves.jl [<filename>...]
"""

args = docopt(doc)

function only(a)
    @assert length(a) == 1
    return first(a)
end

names = Dict("qmdp"=>"QMDP",
             "outcome"=>"Naive MDP",
             "pomcpow"=>"POMCPOW",
             "meanmpc"=>"Mean MPC",
             "baseline"=>"Assume normal",
             "omniscient"=>"Omniscient"
            )

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
    @show sort(cordf, [:solver])
    cor = only(unique(cordf[:cor]))
    plts = Plots.Plot[]
    for df in groupby(cordf, [:solver])
        df = sort(df, [:lambda])
        # if cor == 1.0
        if cor == 0.0 || cor == 0.75 
            name = names[only(unique(df[:solver]))]
        else
            name = nothing
        end
        plt = Plots.Linear(df[:safe], df[:reached_lane],
                           legendentry=name,
                           errorBars=ErrorBars(x=df[:safe_std]./sqrt.(df[:n]),
                                               y=df[:rl_std]./sqrt.(df[:n]))
                          )
        push!(plts, plt)
    end
    a = Axis(plts,
             # title="Correlation $cor",
             ylabel="Fraction successful",
             xlabel="Fraction safe",
             ymin=0.0,
             ymax=1.0,
             # style="legend style={at={(0.5,-0.2)}, anchor=north}, legend columns=2"
             style="legend style={at={(0.5,1.05)}}, font=\\footnotesize, anchor=south}, font=\\footnotesize, legend columns=2, grid=both",
             width="3.5in",
             height="2.7in"
            )

    bd = standard_uniform(correlation=cor)
    N = 500
    bs = [rand(Base.GLOBAL_RNG, bd) for i in 1:N]
    v0s_0 = collect(map(b->b.p_idm.v0, bs))
    Ts_0 = collect(map(b->b.p_idm.T, bs))

    s = Axis(Plots.Scatter(v0s_0, Ts_0; markSize=1);
             width="2in",
             height="2in",
             xlabel="\$T\$ (\\si{\\second})",
             ylabel="\$\\dot{x}_0\$ (\\si{\\meter\\per\\second})",
             title="Parameter distribution slice",
             style="xshift=4in, yshift=0.2in, grid=both",
            )

    file = @sprintf("/home/zach/Devel/thesis/media/gaps_%03d.tex", cor*100)
    # save(file, a)
    save(file, [a,s], include_preamble=false)
    # run(`xdg-open $file`)
end
