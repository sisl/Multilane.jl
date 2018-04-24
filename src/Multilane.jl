__precompile__(true)
module Multilane

import StatsBase: Weights, sample

using POMDPs
import POMDPs: actions, discount, isterminal, iterator
import POMDPs: rand, reward
import POMDPs: solve, action
import POMDPs: update, initialize_belief
import POMDPs: generate_s, generate_sr, initial_state, generate_o, generate_sor

import Distributions: Dirichlet, Exponential, Gamma, rand
import Distributions

import POMDPToolbox: action_info, generate_sori, generate_sri

import Iterators

import Base: ==, hash, length, vec, +, -, *, .*, ^, .^, .-, /, sqrt, zero, abs, max

# import POMDPToolbox: Particle, ParticleBelief

using DataFrames
using ProgressMeter
using PmapProgressMeter
using POMDPToolbox
using MCTS # so that we can define node_tag, etc.
using POMCPOW
using CPUTime
# using RobustMCTS # for RobustMDP
# import POMCP # for particle filter

# using Reel
# using AutomotiveDrivingModels
# using AutoViz

import Mustache
import JLD

using Parameters
using StaticArrays
using SpecialFunctions

include("debug.jl")

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
    ABMDPSolver,
    DiscreteBehaviorSet,
    DiscreteBehaviorBelief,
    WeightUpdateParams,
    UniformIDMMOBIL,
    standard_uniform,
    normalized_error_sum,
    MLMPCSolver,
    MLPOMDPSolver,
    MeanMPCSolver,
    GenQMDPSolver,
    QBSolver,
    QMDPWrapper,
    OutcomeMDP,
    OutcomeSolver

export
    NoCrashRewardModel,
    NoCrashIDMMOBILModel,
    NoCrashMDP,
    NoCrashPOMDP,
    SuccessReward,
    Simple,
    SimpleSolver,
    BehaviorSolver,
    IDMLaneSeekingSolver,
    OptimisticValue

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
    detect_braking,
    max_braking,
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
    run_simulations,
    fill_stats!,
    sbatch_spawn,
    gather_results,
    relaxed_initial_state,
    nan,
    state

export # POMDP belief stuff
    ParticleUpdater,
    create_belief,
    update,
    rand,
    sample,
    initialize_belief,
    BehaviorRootUpdater,
    BehaviorRootUpdaterStub,
    AggressivenessBelief,
    AggressivenessUpdater,
    AggressivenessPOWFilter,
    agg_means,
    agg_stds,
    aggressiveness,
    BehaviorParticleUpdater,
    BehaviorParticleBelief,
    BehaviorPOWFilter,
    ParticleGenerator,
    param_means,
    param_stds

export
    MaxBrakeMetric,
    NumBehaviorBrakesMetric

export
    GaussianCopula

export
    include_visualization

include("sampling.jl")
include("triangular.jl")
include("physical.jl")
include("MDP_types.jl")
include("crash.jl")
include("IDM.jl")
include("MOBIL.jl")
include("behavior.jl")
include("copula.jl")
include("behavior_gen.jl")
include("no_crash_model.jl")
include("success_model.jl")
# include("metrics.jl")
# include("evaluation.jl")
include("test_sets.jl")
include("relax.jl")
include("robust_mdp.jl")
include("pomdp_glue.jl")
include("beliefs.jl")
include("aggressiveness_particle_filter.jl")
include("uniform_particle_filter.jl")
include("pow_filter.jl")
include("most_likely_mpc.jl")
include("aggressiveness_belief_mdp.jl")
include("qmdp.jl")
include("qbmcts.jl")
include("outcome_uncertainty.jl")
include("single_behavior.jl")
include("heuristics.jl")
include("tree_vis.jl")
# include("sherlock.jl")

include_visualization() = include(joinpath(Pkg.dir("Multilane"),"src","visualization.jl"))

if gethostname() == "Theresa"
    println("Automatically loading visualization components.")
    include("visualization.jl")
    export
        interp_state,
        visualize,
        make_rollouts,
        NodeWithRollouts
    # export
    #     display_sim,
    #     show_state,
    #     show_sim,
    #     display_sim,
    #     save_frame,
    #     visualize
end

end # module
