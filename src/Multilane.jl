module Multilane

import StatsBase: WeightVec, sample

using POMDPs
import POMDPs: actions, discount, isterminal, iterator
import POMDPs: create_action, create_state, rand, reward, create_observation
import POMDPs: solve, action
import POMDPs: create_belief, update, initialize_belief
using GenerativeModels
import GenerativeModels: generate_s, generate_sr, initial_state, generate_o, generate_sor

import Distributions: Dirichlet, Exponential, Gamma, rand

import Iterators.product

import Base: ==, hash, length, vec

import POMDPToolbox: Particle, ParticleBelief

# for visualization
using Interact

# package code goes here
export
    PhysicalParam,
    CarState,
    MLState,
    MLAction,
    MLObs,
    BehaviorModel,
    MLMDP,
    MLPOMDP,
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
    NoCrashPOMDP,
    Simple,
    SimpleSolver

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

export #data structure stuff
    NoCrashStat,
    NoCrashStats,
    mean,
    std,
    ste,
    test_run

export # POMDP belief stuff
    ParticleUpdater,
    create_belief,
    update,
    rand,
    sample,
    initialize_belief


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
include("heuristics.jl")
include("beliefs.jl")

end # module
