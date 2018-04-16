mutable struct AggressivenessBelief <: BehaviorBelief
    gen::CorrelatedIDMMOBIL
    physical::MLPhysicalState
    particles::Vector{Vector{Float64}} # First index is the position in physical.cars
    cweights::Vector{Vector{Float64}}   # Second index is the particle number 
end

"""
Return the weights for the behavior particles of a car (as opposed to the cumulative weights).
"""
weights(b::AggressivenessBelief, i::Int) = insert!(diff(b.cweights[i]), 1, first(b.cweights[i]))

function rand(rng::AbstractRNG,
                 b::AggressivenessBelief,
                 s::MLState=MLState(b.physical, Vector{CarState}(length(b.physical.cars))))
    return rand(rng, b, zeros(length(b.physical.cars)), s)
end

function rand(rng::AbstractRNG,
                b::AggressivenessBelief,
                sample_noises::Vector{Float64},
                s::MLState=MLState(b.physical, Vector{CarState}(length(b.physical.cars))))

    s.x = b.physical.x
    s.t = b.physical.t
    resize!(s.cars, length(b.physical.cars))
    for i in 1:length(s.cars) # XXX could speed this up by sampling all at once
        particle = sample_cweighted(rng, b.particles[i], b.cweights[i])
        nudged = max(min(particle+sample_noises[i]*randn(rng),1.0),0.0)
        @assert !isnan(nudged)
        m = create_model(b.gen, nudged)
        @assert !any(isnan, m.p_idm)
        s.cars[i] = CarState(b.physical.cars[i], m)
    end
    s.terminal = b.physical.terminal
    return s
end

actions(p::Union{MLMDP,MLPOMDP}, b::AggressivenessBelief) = actions(p, b.physical)

function most_likely_state(b::AggressivenessBelief)
    s = MLState(b.physical, Vector{CarState}(length(b.physical.cars)))
    for i in 1:length(s.cars)
        ml_ind = indmax(weights(b, i))
        behavior = create_model(b.gen, b.particles[i][ml_ind])
        s.cars[i] = CarState(b.physical.cars[i], behavior)
    end
    return s
end
agg_means(b::AggressivenessBelief) = [mean(b.particles[i], Weights(weights(b, i))) for i in 1:length(b.particles)]
function agg_stds(b::AggressivenessBelief)
    means = agg_means(b)
    stds = Vector{Float64}(length(b.physical.cars))
    for i in 1:length(b.physical.cars)
        wts = weights(b, i)
        stds[i] = sqrt(sum(wts.*(b.particles[i]-means[i]).^2)/sum(wts))
    end
    return stds
end

"""
Return a vector of states sampled using a carwise version of Thrun's Probabilistic Robotics p. 101
"""
function lv_resample(b, up)
    rng = up.rng
    stds = max.(agg_stds(b, 0.01))
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
                particle = clamp(particle+stds[ci]*randn(rng), 0.0, 1.0)
            end
            cs = CarState(b.physical.cars[ci], create_model(b.gen, particle))
            samples[inds[m]].cars[ci] = cs
        end
    end
    return samples
end

function weights_from_particles!(b::AggressivenessBelief,
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
            b.particles[i] = Vector{Float64}(length(particles))
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

    @if_debug begin
        if any(ws->any(isnan,ws), b.cweights)
            warn("NaN weight in aggressiveness filter.")
        end
        if any(ws->any(isnan,ws), b.cweights)
            warn("NaN particle in aggressiveness filter.")
        end
    end

    for cw in b.cweights
        if sum(cw) == 0.0
            cw[:] = 1.0:1.0:length(cw)
        end
    end
   
    return b
end

function maybe_push_one!(particles::Vector{Vector{Float64}}, cweights, params, pp, gen, sp, o)
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
                        push!(particles[io], aggressiveness(gen, csp.behavior))
                        push!(cweights[io], cweight + proportional_likelihood)
                    elseif abs(co.y - csp.y) <= 1.1
                        push!(particles[io], aggressiveness(gen, csp.behavior))
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

mutable struct AggressivenessUpdater <: Updater
    problem::Nullable{NoCrashProblem}
    nb_sims::Int
    p_resample_noise::Float64
    resample_noise_factor::Float64 
    params::WeightUpdateParams
    rng::AbstractRNG
end
function set_problem!(u::AggressivenessUpdater, p::Union{POMDP,MDP})
    u.problem = Nullable{NoCrashProblem}(p)
end
function set_rng!(u::AggressivenessUpdater, rng::AbstractRNG)
    u.rng = rng
end

function update(up::AggressivenessUpdater,
                b_old::AggressivenessBelief,
                a::MLAction,
                o::MLPhysicalState)

    b_new = AggressivenessBelief(CorrelatedIDMMOBIL(
                                 get(up.problem).dmodel.behaviors), o,
                                 Vector{Vector{Float64}}(length(o.cars)),
                                 Vector{Vector{Float64}}(length(o.cars)))

    particles = Vector{MLState}(up.nb_sims)
    samples = lv_resample(b_old, up)
    for i in 1:up.nb_sims
        particles[i] = generate_s(get(up.problem), samples[i], a, up.rng)
    end
    
    weights_from_particles!(b_new, get(up.problem), o, particles, up.params)

    for i in 1:length(o.cars)
        if isempty(b_new.cweights[i])
            # println("car $i has empty weights")
            b_new.particles[i] = rand(up.rng, up.nb_sims)
            b_new.cweights[i] = 1.0:1.0:up.nb_sims
        end
    end

    return b_new
end

function initialize_belief(up::AggressivenessUpdater, distribution)
    gen = CorrelatedIDMMOBIL(get(up.problem).dmodel.behaviors)
    states = [rand(up.rng, distribution) for i in 1:up.nb_sims]
    particles = Vector{Vector{Float64}}(length(first(states).cars))
    cweights = Vector{Vector{Float64}}(length(first(states).cars))
    for i in 1:length(first(states).cars)
        particles[i] = Vector{Float64}(length(states))
        cweights[i] = 1.0:1.0:length(states)
        for (j,s) in enumerate(states)
            particles[i][j] = aggressiveness(gen, s.cars[i].behavior)
        end
    end
    return AggressivenessBelief(gen, MLPhysicalState(first(states)), particles, cweights)
end

function initialize_belief(up::AggressivenessUpdater, physical::MLPhysicalState)
    gen = CorrelatedIDMMOBIL(get(up.problem).dmodel.behaviors)
    particles = Vector{Vector{Float64}}(length(physical.cars))
    cweights = Vector{Vector{Float64}}(length(physical.cars))
    for i in 1:length(physical.cars)
        particles[i] = Vector{Float64}(up.nb_sims)
        cweights[i] = 1.0:1.0:up.nb_sims
        for j in 1:up.nb_sims
            particles[i][j] = aggressiveness(gen, rand(up.rng, gen))
        end
    end
    return AggressivenessBelief(gen, physical, particles, cweights)
end

initialize_belief(up::AggressivenessUpdater, d::AggressivenessBelief) = d
