mutable struct BehaviorParticleBelief{G} <: BehaviorBelief
    gen::G
    physical::MLPhysicalState
    particles::Vector{Vector{IDMMOBILBehavior}} # First index is the position in physical.cars
    cweights::Vector{Vector{Float64}}   # Cumulative weights Second index is the particle number
end

"""
Return the weights for the behavior particles of a car (as opposed to the cumulative weights).
"""
weights(b::BehaviorParticleBelief, i::Int) = insert!(diff(b.cweights[i]), 1, first(b.cweights[i]))

function rand(rng::AbstractRNG,
                b::BehaviorParticleBelief,
                s::MLState=MLState(b.physical, Vector{CarState}(length(b.physical.cars))))
    resize!(s.cars, length(b.physical.cars))
    for i in 1:length(s.cars)
        particle = sample_cweighted(rng, b.particles[i], b.cweights[i])
        s.cars[i] = CarState(b.physical.cars[i], particle)
    end
    return s
end

function rand(rng::AbstractRNG,
                b::BehaviorParticleBelief,
                sample_noises::Vector)

    s::MLState=MLState(b.physical, Vector{CarState}(length(b.physical.cars)))
    resize!(s.cars, length(b.physical.cars))
    for i in 1:length(s.cars)
        particle = sample_cweighted(rng, b.particles[i], b.cweights[i])
        nudged = clip(particle+sample_noises[i].*randn(rng,9), b.gen)
        s.cars[i] = CarState(b.physical.cars[i], nudged)
    end
    return s
end

# action(p::Policy, b::BehaviorParticleBelief) = action(p, b.physical)

function most_likely_state(b::BehaviorParticleBelief)
    s = MLState(b.physical, Vector{CarState}(length(b.physical.cars)))
    for i in 1:length(s.cars)
        ml_ind = indmax(weights(b, i))
        behavior = b.particles[i][ml_ind]
        s.cars[i] = CarState(b.physical.cars[i], behavior)
    end
    return s
end

function Base.mean(b::BehaviorParticleBelief)
    bs = param_means(b)
    pcs = b.physical.cars
    cars = [CarState(pcs[i], bs[i]) for i in 1:length(bs)]
    return MLState(b.physical, cars)
end

function param_means(b::BehaviorParticleBelief)
    ms = Vector{eltype(first(b.particles))}(length(b.physical.cars))
    for i in 1:length(ms)
        wts = weights(b, i)
        ms[i] = sum(wts.*b.particles[i])/sum(wts)
    end
    return ms
end
function param_stds(b::BehaviorParticleBelief)
    means = param_means(b)
    stds = Vector{IDMMOBILBehavior}(length(b.physical.cars))
    for i in 1:length(b.physical.cars)
        wts = weights(b, i)
        stds[i] = sqrt(sum(wts.*(b.particles[i].-means[i]).^2)/sum(wts))
    end
    return stds
end


function cweights_from_particles!(b::BehaviorParticleBelief,
                                 problem::NoCrashProblem,
                                 o::MLPhysicalState,
                                 particles,
                                 p::WeightUpdateParams)

    b.physical = o
    resize!(b.cweights, length(o.cars))
    resize!(b.particles, length(o.cars))
    for i in 1:length(o.cars)
        # make sure we're not going to be allocating a bunch of memory in the loop
        if isassigned(b.particles, i)
            sizehint!(b.particles[i], length(particles))
            resize!(b.particles[i], 0)
        else
            b.particles[i] = Vector{IDMMOBILBehavior}(length(particles))
            resize!(b.particles[i], 0)
        end
        if isassigned(b.cweights, i)
            sizehint!(b.cweights[i], length(particles))
            resize!(b.cweights[i], 0)
        else
            b.cweights[i] = Vector{Float64}(length(particles))
            resize!(b.cweights[i], 0)
        end
    end
    for sp in particles
        maybe_push_one!(b.particles, b.cweights, p, problem.dmodel.phys_param, b.gen, sp, o)
    end
   
    return b
end

function maybe_push_one!(particles::Vector{Vector{IDMMOBILBehavior}}, cweights, params, pp, gen, sp, o)
    isp = 1
    io = 1
    while io <= length(o.cars) && isp <= length(sp.cars)
        co = o.cars[io]
        csp = sp.cars[isp]
        if co.id == csp.id
            if abs(co.x-csp.x) < 0.2*pp.lane_length
                @assert isa(csp.behavior, IDMMOBILBehavior)
                a = csp.behavior.p_idm.a
                dt = pp.dt
                veld = TriangularDist(csp.vel-a*dt/2.0, csp.vel+a*dt/2.0, csp.vel)
                proportional_likelihood = Distributions.pdf(veld, co.vel)
                if proportional_likelihood > 0.0
                    cweight = length(cweights[io]) > 0 ? last(cweights[io]) : 0.0
                    if co.y == csp.y
                        push!(particles[io], csp.behavior)
                        push!(cweights[io], cweight + proportional_likelihood)
                    elseif abs(co.y - csp.y) < 1.0
                        push!(particles[io], csp.behavior)
                        push!(cweights[io], cweight + params.wrong_lane_factor*proportional_likelihood)
                    end # if greater than one lane apart, do nothing
                end
            end
            io += 1
            isp += 1
        elseif co.id < csp.id
            io += 1
        else 
            @assert co.id > csp.id
            isp += 1
        end
    end
end

actions(p::Union{MLMDP,MLPOMDP}, b::BehaviorParticleBelief) = actions(p, b.physical)

mutable struct BehaviorParticleUpdater <: Updater
    problem::Nullable{NoCrashProblem}
    nb_sims::Int
    p_resample_noise::Float64
    resample_noise_factor::Float64 
    params::WeightUpdateParams
    rng::AbstractRNG
end
function set_problem!(u::BehaviorParticleUpdater, p::Union{POMDP,MDP})
    u.problem = Nullable{NoCrashProblem}(p)
end
function set_rng!(u::BehaviorParticleUpdater, rng::AbstractRNG)
    u.rng = rng
end

"""
Return a vector of states sampled using a carwise version of Thrun's Probabilistic Robotics p. 101
"""
function lv_resample(b::BehaviorParticleBelief, up::BehaviorParticleUpdater)
    n = up.nb_sims
    rng = up.rng
    gen = b.gen
    min_std = 0.001*IDMMOBILBehavior(gen.max_idm-gen.min_idm, gen.max_mobil-gen.min_mobil, 0)
    stds = max.(param_stds(b), min_std)
    samples = Array{MLState}(n)
    nc = length(b.physical.cars)
    for m in 1:n
        cars = resize!([CarState(first(b.physical.cars), NORMAL)], nc)
        samples[m] = MLState(b.physical, cars)
    end
    for ci in 2:nc
        inds = randperm(rng, n)
        particles = b.particles[ci]
        cweights = b.cweights[ci]
        step = last(cweights)/n
        r = rand(rng)*step
        c = first(cweights)
        i = 1
        U = r
        for m in 1:n
            while U > c
                i += 1
                c = cweights[i]
            end
            U += step
            particle = particles[i]
            if rand(up.rng) < up.p_resample_noise
                particle = clip(particle+stds[ci].*randn(rng, 9), gen)
            end
            cs = CarState(b.physical.cars[ci], particle)
            samples[inds[m]].cars[ci] = cs
        end
    end
    return samples
end


function update(up::BehaviorParticleUpdater,
                b_old::BehaviorParticleBelief,
                a::MLAction,
                o::MLPhysicalState,
                b_new::BehaviorParticleBelief=BehaviorParticleBelief(get(up.problem).dmodel.behaviors, o,
                                                    Vector{Vector{IDMMOBILBehavior}}(length(o.cars)),
                                                    Vector{Vector{Float64}}(length(o.cars))))

    gen = get(up.problem).dmodel.behaviors
    particles = Vector{MLState}(up.nb_sims)
    samples = lv_resample(b_old, up)
    for i in 1:up.nb_sims
        particles[i] = generate_s(get(up.problem), samples[i], a, up.rng)
    end
    
    cweights_from_particles!(b_new, get(up.problem), o, particles, up.params)

    for i in 1:length(o.cars)
        if isempty(b_new.cweights[i])
            b_new.particles[i] = [rand(up.rng, b_new.gen) for i in 1:up.nb_sims]
            b_new.cweights[i] = collect(1.0:convert(Float64, up.nb_sims))./up.nb_sims
        end
    end

    return b_new
end

function initialize_belief(up::BehaviorParticleUpdater, distribution)
    gen = get(up.problem).dmodel.behaviors
    states = [rand(up.rng, distribution) for i in 1:up.nb_sims]
    particles = Vector{Vector{IDMMOBILBehavior}}(length(first(states).cars))
    weights = Vector{Vector{Float64}}(length(first(states).cars))
    for i in 1:length(first(states).cars)
        particles[i] = Vector{IDMMOBILBehavior}(length(states))
        weights[i] = collect(1.0:convert(Float64, up.nb_sims))./up.nb_sims
        for (j,s) in enumerate(states)
            particles[i][j] = s.cars[i].behavior
        end
    end
    return BehaviorParticleBelief(gen, MLPhysicalState(first(states)), particles, weights)
end

function initialize_belief(up::BehaviorParticleUpdater, physical::MLPhysicalState)
    gen = get(up.problem).dmodel.behaviors
    particles = Vector{Vector{IDMMOBILBehavior}}(length(physical.cars))
    weights = Vector{Vector{Float64}}(length(physical.cars))
    for i in 1:length(physical.cars)
        particles[i] = Vector{IDMMOBILBehavior}(up.nb_sims)
        weights[i] = collect(1.0:convert(Float64, up.nb_sims))./up.nb_sims
        for j in 1:up.nb_sims
            particles[i][j] = rand(up.rng, gen)
        end
    end
    return BehaviorParticleBelief(gen, physical, particles, weights)
end
