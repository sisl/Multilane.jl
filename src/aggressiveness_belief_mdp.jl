struct AggressivenessBeliefMDP <: MDP{AggressivenessBelief, MLAction}
    up::AggressivenessUpdater
end

function generate_sr(p::AggressivenessBeliefMDP, b_old::AggressivenessBelief, a::MLAction, rng::AbstractRNG)
    up = p.up
    pomdp = get(up.problem)
    s = rand(rng, b_old)
    sp = generate_s(pomdp, s, a, rng)
    o = generate_o(pomdp, s, a, sp)

    b_new = AggressivenessBelief(CorrelatedIDMMOBIL(
                                 get(up.problem).dmodel.behaviors), o,
                                 Vector{Vector{Float64}}(length(o.cars)),
                                 Vector{Vector{Float64}}(length(o.cars)))

    rsum = 0.0
    particles = Vector{MLState}(up.nb_sims)
    stds = max.(agg_stds(b_old), 0.01)
    for i in 1:up.nb_sims
        if rand(up.rng) < up.p_resample_noise
            s = rand(up.rng, b_old, up.resample_noise_factor*stds)
        else
            s = rand(up.rng, b_old)
        end
        particles[i], r = generate_sr(get(up.problem), s, a, up.rng)
        rsum += r
    end
    
    cweights_from_particles!(b_new, get(up.problem), o, particles, up.params)

    for i in 1:length(o.cars)
        if isempty(b_new.cweights[i])
            # println("car $i has empty weights")
            b_new.particles[i] = rand(up.rng, up.nb_sims)
            b_new.cweights[i] = 1.0:1.0:up.nb_sims
        end
    end

    return b_new, rsum/up.nb_sims
end

# actions(p::AggressivenessBeliefMDP) = actions(get(p.up.problem))
actions(p::AggressivenessBeliefMDP, b::AggressivenessBelief) = actions(get(p.up.problem), b.physical)
discount(p::AggressivenessBeliefMDP) = discount(get(p.up.problem))

struct ABMDPSolver <: Solver
    solver
    updater
end

function solve(sol::ABMDPSolver, pomdp)
    up = deepcopy(sol.updater)
    set_problem!(up, pomdp)
    return solve(sol.solver, AggressivenessBeliefMDP(up))
end

struct BehaviorBeliefMDP{G} <: MDP{BehaviorParticleBelief{G}, MLAction}
    up::BehaviorParticleUpdater
end

BehaviorBeliefMDP(up) = BehaviorBeliefMDP{typeof(get(up.problem).dmodel.behaviors)}(up)

function generate_sr(p::BehaviorBeliefMDP, b_old::BehaviorParticleBelief, a::MLAction, rng::AbstractRNG)
    up = p.up
    pomdp = get(up.problem)
    s = rand(rng, b_old)
    sp = generate_s(pomdp, s, a, rng)
    o = generate_o(pomdp, s, a, sp)


    b_new::BehaviorParticleBelief=BehaviorParticleBelief(get(up.problem).dmodel.behaviors, o,
                                        Vector{Vector{IDMMOBILBehavior}}(length(o.cars)),
                                        Vector{Vector{Float64}}(length(o.cars)))


    rsum = 0.0
    particles = Vector{MLState}(up.nb_sims)
    samples = lv_resample(b_old, up)
    for i in 1:up.nb_sims
        particles[i], r = generate_sr(get(up.problem), samples[i], a, up.rng)
        rsum += r
    end
    
    cweights_from_particles!(b_new, get(up.problem), o, particles, up.params)

    for i in 1:length(o.cars)
        if isempty(b_new.cweights[i])
            b_new.particles[i] = [rand(up.rng, b_new.gen) for i in 1:up.nb_sims]
            b_new.cweights[i] = collect(1.0:convert(Float64, up.nb_sims))./up.nb_sims
        end
    end

    return b_new, rsum/up.nb_sims
end

actions(p::BehaviorBeliefMDP, b::BehaviorParticleBelief) = actions(get(p.up.problem), b.physical)
discount(p::BehaviorBeliefMDP) = discount(get(p.up.problem))

struct BBMDPSolver <: Solver
    solver
    updater
end

function solve(sol::BBMDPSolver, pomdp)
    up = deepcopy(sol.updater)
    set_problem!(up, pomdp)
    return solve(sol.solver, BehaviorBeliefMDP(up))
end
