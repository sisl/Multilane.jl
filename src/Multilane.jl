__precompile__()
module Multilane

import StatsBase: WeightVec, sample

using POMDPs
import POMDPs: actions, discount, isterminal, iterator
import POMDPs: create_action, create_state, rand, reward, create_observation, create_policy
import POMDPs: solve, action
import POMDPs: create_belief, update, initialize_belief
using GenerativeModels
import GenerativeModels: generate_s, generate_sr, initial_state, generate_o, generate_sor

import Distributions: Dirichlet, Exponential, Gamma, rand

import Iterators

import Base: ==, hash, length, vec

# import POMDPToolbox: Particle, ParticleBelief

# for visualization
using Interact

using DataFrames
using ProgressMeter
using PmapProgressMeter
using POMDPToolbox
import MCTS # so that we can define node_tag, etc.
using RobustMCTS # for RobustMDP
import POMCP # for particle filter

using Reel
using AutomotiveDrivingModels
using AutoViz


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
    IDMMOBILBehavior,
    MLPhysicalState,
    CarPhysicalState,
    RobustNoCrashMDP,
    FixedBehaviorNoCrashMDP,
    StochasticBehaviorNoCrashMDP,
    RobustMLSolver,
    RobustMLPolicy,
    MLPOMDPSolver,
    MLPOMDPAgent,
    BehaviorRootUpdater,
    BehaviorRootUpdaterStub,
    DiscreteBehaviorSet,
    DiscreteBehaviorBelief

export
    NoCrashRewardModel,
    NoCrashIDMMOBILModel,
    NoCrashMDP,
    NoCrashPOMDP,
    Simple,
    SimpleSolver,
    BehaviorSolver


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
    display_sim,
    write_tmp_gif,
    detect_braking,
    braking_ids

export #data structure stuff
    ste,
    test_run,
    evaluate,
    merge_results!,
    rerun,
    assign_keys,
    test_run_return_policy,
    lambda,
    TestSet,
    gen_initials,
    run_simulations


export # POMDP belief stuff
    ParticleUpdater,
    create_belief,
    update,
    rand,
    sample,
    initialize_belief

export
    MaxBrakeMetric,
    NumBehaviorBrakesMetric


include("physical.jl")
include("MDP_types.jl")
include("crash.jl")
include("IDM.jl")
include("MOBIL.jl")
include("behavior.jl")
include("behavior_gen.jl")
include("no_crash_model.jl")
include("visualization.jl")
include("metrics.jl")
include("evaluation.jl")
include("test_sets.jl")
include("robust_mdp.jl")
include("pomdp_glue.jl")
include("single_behavior.jl")
include("heuristics.jl")
include("beliefs.jl")
include("tree_vis.jl")

end # module
