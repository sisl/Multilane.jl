push!(LOAD_PATH,joinpath("..","src"))
using Multilane
using JLD

# NOTE: assumes 4 lanes, desired lane is lane 4
desired_lane_reward = 10.

lambdas = Float64[0.1, 0.3, 0.5, 1., 3., 5., 10., 30., 50., 100.]

nb_lanes = 4 # XXX assumption

rmodels = Multilane.NoCrashRewardModel[Multilane.NoCrashRewardModel(desired_lane_reward*lambda,desired_lane_reward,1.5,nb_lanes) for lambda in lambdas]

jldopen(joinpath(".","rmodels.jld"),"w") do file
  addrequire(file,Multilane)
  write(file,"rmodels",rmodels)
end
