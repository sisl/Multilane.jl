using Multilane
using MCTS
using POMCP

# dir = "/tmp/sim_data_Aug_22_19_22"
dir = "/scratch/users/zsunberg/sim_data_Aug_22_19_51/"

files = []

N = 10
for i in 1:N
    push!(files, joinpath(dir, "results_$(i)_of_$(N).jld"))
end

tic()
results = @time gather_results(files, save_file=Nullable(string("results_, ", Dates.format(Dates.now(),"u_d_HH_MM"), ".jld")))
toc()
