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
    @if_debug if any(isnan, stds)
        Gallium.@enter update(up, b_old, a, o)
    end
    for i in 1:up.nb_sims
        if rand(up.rng) < up.p_resample_noise
            s = rand(up.rng, b_old, up.resample_noise_factor*stds)
        else
            s = rand(up.rng, b_old)
        end
        particles[i], r = generate_sr(get(up.problem), s, a, up.rng)
        rsum += r
    end
    
    weights_from_particles!(b_new, get(up.problem), o, particles, up.params)

    for i in 1:length(o.cars)
        if isempty(b_new.weights[i])
            b_new.particles[i] = rand(up.rng, up.nb_sims)
            b_new.weights[i] = ones(up.nb_sims)
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
