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

using DataFrames
using ProgressMeter
using PmapProgressMeter
using POMDPToolbox
import MCTS # so that we can define node_tag, etc.

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
    SingleBehaviorPolicy,
    single_behavior_state

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
    ste,
    test_run,
    evaluate,
    merge_results!,
    rerun,
    assign_keys


export # POMDP belief stuff
    ParticleUpdater,
    create_belief,
    update,
    rand,
    sample,
    initialize_belief


include("util.jl")
include("physical.jl")
include("MDP_types.jl")
include("crash.jl")
# include("MDP.jl")
include("IDM.jl")
include("MOBIL.jl")
include("behavior.jl")
include("no_crash_model.jl")
include("visualization.jl")
include("evaluation.jl")
include("single_behavior.jl")
include("heuristics.jl")
include("beliefs.jl")
include("tree_vis.jl")

end # module
