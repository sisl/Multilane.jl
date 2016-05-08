module Multilane

import StatsBase: WeightVec, sample

using POMDPs
import POMDPs: actions, discount, isterminal, iterator
import POMDPs: create_action, create_state, rand, reward
import POMDPs: solve, action
using GenerativeModels
import GenerativeModels: generate_s, generate_sr, initial_state

import Distributions: Dirichlet, Exponential, Gamma, rand

import Iterators.product

import Base: ==, hash, length, vec

# for visualization
using Interact

# package code goes here
export
    PhysicalParam,
    CarState,
    MLState,
    MLAction,
    BehaviorModel,
    MLMDP,
    OriginalMDP,
    OriginalRewardModel,
    IDMMOBILModel,
    IDMParam,
    MOBILParam,
    IDMMOBILBehavior

export
    NoCrashRewardModel,
    NoCrashIDMMOBILModel,
    NoCrashMDP,
    Simple

export
    SingleBehaviorSolver,
    SingleBehaviorPolicy

export
    get_neighborhood, #testing VVV
    get_dv_ds,
    is_lanechange_dangerous,
    get_rear_accel, #testing ^^^
    get_idm_dv,
    get_mobil_lane_change,
    is_crash,
    visualize,
    display_sim


include("physical.jl")
include("MDP_types.jl")
include("crash.jl")
include("MDP.jl")
include("IDM.jl")
include("MOBIL.jl")
include("behavior.jl")
include("no_crash_model.jl")
include("visualization.jl")
include("evaluation.jl")
include("single_behavior.jl")

end # module
