using Multilane
using MCTS
using POMCP

dir = "/tmp/sim_data_Aug_22_19_22"

files = []

N = 2
for i in 1:N
    push!(files, joinpath(dir, "results_$(i)_of_$(N).jld"))
end

results = gather_results(files, save_file=Nullable(string("results_, ", Dates.format(Dates.now(),"u_d_HH_MM"), ".jld")))
