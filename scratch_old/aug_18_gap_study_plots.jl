using Multilane
using JLD
using Multilane
using POMDPs
using MCTS
using DataFrames
using DataFramesMeta
using Plots
using StatPlots


# results = load("results_Aug_22_23_26.jld")
# results = load("combined_results_Aug_25_10_10.jld")
# results = load("combined_results_Aug_26_19_53.jld")
# results = load("combined_results_Aug_27_14_05.jld")
results = load("combined_results_Aug_29_10_08.jld")

stats = results["stats"]
tests = results["tests"]

mean_performance = by(stats, [:lambda, :test_key]) do df
    DataFrame(steps_in_lane=mean(df[:steps_in_lane]),
              nb_brakes=mean(df[:nb_brakes]),
              time_to_lane=mean(df[:time_to_lane]),
              steps=mean(df[:steps]),
              test_key=first(df[:test_key])
              )
end
mean_performance[:brakes_per_sec] = mean_performance[:nb_brakes]./mean_performance[:time_to_lane]

plts = []
# for p in linspace(0., 1., 5)
# for p in linspace(0., 3/4, 4)
for p in linspace(0., 1.25, 6)
    ubg = @where(mean_performance, :test_key.==@sprintf("upper_bound_%03d", 100*p))
    plot(ubg, :time_to_lane, :nb_brakes, group=:test_key)
    ang = @where(mean_performance, :test_key.==@sprintf("assume_normal_%03d", 100*p))
    push!(plts, plot!(ang, :time_to_lane, :nb_brakes, group=:test_key, title="Parameter Range = $p", xlim=(0,25), ylim=(0,2.5)))
end
plot(plts...)
gui()
