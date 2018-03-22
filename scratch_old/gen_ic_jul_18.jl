using Multilane
using GenerativeModels
using JLD

N = 5000

# NOTE: assumes 4 lanes, desired lane is lane 4
desired_lane_reward = 10.

lambdas = Float64[0.1, 1., 10., 17.78, 31.62, 56.23, 100., 1000.]

nb_lanes = 4 # XXX assumption

rmodels = Multilane.NoCrashRewardModel[
                Multilane.NoCrashRewardModel(desired_lane_reward*lambda,
                                             desired_lane_reward,2.5,nb_lanes)
                for lambda in lambdas]

nb_lanes = 4
pp = PhysicalParam(nb_lanes,lane_length=100.) #2.=>col_length=8\n",
_discount = 1.0
nb_cars=10
dmodel = NoCrashIDMMOBILModel(nb_cars, pp, lane_terminate=false)

curve_problems = []
for i in 1:length(lambdas)
    rmodel = rmodels[i]
    mdp = NoCrashMDP(dmodel, rmodel, _discount);
    push!(curve_problems, mdp)
end

isrng = MersenneTwister(123)
initial_states = MLState[initial_state(curve_problems[1], isrng) for i in 1:N]

initials = assign_keys(curve_problems, initial_states, rng=isrng)

filename = string("initials_", Dates.format(Dates.now(),"u_d_HH_MM"), ".jld")
save(filename, initials)
println("problems and initial conditions saved to $filename")
