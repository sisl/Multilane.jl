mutable struct AggressivenessBelief <: BehaviorBelief
    gen::CorrelatedIDMMOBIL
    physical::MLPhysicalState
    particles::Vector{Vector{Float64}} # First index is the position in physical.cars
    weights::Vector{Vector{Float64}}   # Second index is the particle number
end

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
    for i in 1:length(s.cars)
        particle = sample(rng, b.particles[i], Weights(b.weights[i]))
        nudged = max(min(particle+sample_noises[i]*randn(rng),1.0),0.0)
        s.cars[i] = CarState(b.physical.cars[i], create_model(b.gen, nudged))
    end
    s.terminal = b.physical.terminal
    return s
end

# action(p::Policy, b::AggressivenessBelief) = action(p, b.physical)

function most_likely_state(b::AggressivenessBelief)
    s = MLState(b.physical, Vector{CarState}(length(b.physical.cars)))
    for i in 1:length(s.cars)
        ml_ind = indmax(b.weights[i])
        behavior = create_model(b.gen, b.particles[i][ml_ind])
        s.cars[i] = CarState(b.physical.cars[i], behavior)
    end
    return s
end
agg_means(b::AggressivenessBelief) = [mean(b.particles[i], Weights(b.weights[i])) for i in 1:length(b.particles)]
function agg_stds(b::AggressivenessBelief)
    means = agg_means(b)
    stds = Vector{Float64}(length(b.physical.cars))
    for i in 1:length(b.physical.cars)
        stds[i] = sqrt(sum(b.weights[i].*(b.particles[i]-means[i]).^2)/sum(b.weights[i]))
    end
    return stds
end


function weights_from_particles!(b::AggressivenessBelief,
                                 problem::NoCrashProblem,
                                 o::MLPhysicalState,
                                 particles,
                                 p::WeightUpdateParams)

    b.physical = o
    resize!(b.weights, length(o.cars))
    resize!(b.particles, length(o.cars))
    for i in 1:length(o.cars)
        # make sure we're not going to be allocating a bunch of memory in the loop
        if isdefined(b.particles, i)
            sizehint!(b.particles[i], length(particles))
            resize!(b.particles[i], 0)
        else
            b.particles[i] = Vector{Float64}(length(particles))
            resize!(b.particles[i], 0)
        end
        if isdefined(b.weights, i)
            sizehint!(b.weights[i], length(particles))
            resize!(b.weights[i], 0)
        else
            b.weights[i] = Vector{Float64}(length(particles))
            resize!(b.weights[i], 0)
        end
    end
    for (j, sp) in enumerate(particles)
        isp = 1
        io = 1
        while io <= length(o.cars) && isp <= length(sp.cars)
            co = o.cars[io]
            csp = sp.cars[isp]
            if co.id == csp.id
                if abs(co.x-csp.x) < 0.2*problem.dmodel.phys_param.lane_length
                    @assert isa(csp.behavior, IDMMOBILBehavior)
                    a = csp.behavior.p_idm.a
                    dt = problem.dmodel.phys_param.dt
                    veld = TriangularDist(csp.vel-a*dt/2.0, csp.vel+a*dt/2.0, csp.vel)
                    proportional_likelihood = Distributions.pdf(veld, co.vel)
                    if co.y == csp.y
                        push!(b.particles[io], aggressiveness(b.gen, csp.behavior))
                        push!(b.weights[io], proportional_likelihood)
                    elseif abs(co.y - csp.y) <= 1.1
                        push!(b.particles[io], aggressiveness(b.gen, csp.behavior))
                        push!(b.weights[io], p.wrong_lane_factor*proportional_likelihood)
                    end # if greater than one lane apart, do nothing
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
   
    return b
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
                o::MLPhysicalState,
                b_new::AggressivenessBelief=AggressivenessBelief(CorrelatedIDMMOBIL(
                                                    get(up.problem).dmodel.behaviors), o,
                                                    Vector{Vector{Float64}}(length(o.cars)),
                                                    Vector{Vector{Float64}}(length(o.cars))))

    particles = Vector{MLState}(up.nb_sims)
    stds = max(agg_stds(b_old), 0.01)
    for i in 1:up.nb_sims
        if rand(up.rng) < up.p_resample_noise
            s = rand(up.rng, b_old, up.resample_noise_factor*stds)
        else
            s = rand(up.rng, b_old)
        end
        particles[i] = generate_s(get(up.problem), s, a, up.rng)
    end
    
    weights_from_particles!(b_new, get(up.problem), o, particles, up.params)

    for i in 1:length(o.cars)
        if isempty(b_new.weights[i])
            # println("car $i has empty weights")
            b_new.particles[i] = rand(up.rng, up.nb_sims)
            b_new.weights[i] = ones(up.nb_sims)
        end
    end

    return b_new
end

function initialize_belief(up::AggressivenessUpdater, distribution)
    gen = CorrelatedIDMMOBIL(get(up.problem).dmodel.behaviors)
    states = [rand(up.rng, distribution) for i in 1:up.nb_sims]
    particles = Vector{Vector{Float64}}(length(first(states).cars))
    weights = Vector{Vector{Float64}}(length(first(states).cars))
    for i in 1:length(first(states).cars)
        particles[i] = Vector{Float64}(length(states))
        weights[i] = ones(length(states))
        for (j,s) in enumerate(states)
            particles[i][j] = aggressiveness(gen, s.cars[i].behavior)
        end
    end
    return AggressivenessBelief(gen, MLPhysicalState(first(states)), particles, weights)
end

function initialize_belief(up::AggressivenessUpdater, physical::MLPhysicalState)
    gen = CorrelatedIDMMOBIL(get(up.problem).dmodel.behaviors)
    particles = Vector{Vector{Float64}}(length(physical.cars))
    weights = Vector{Vector{Float64}}(length(physical.cars))
    for i in 1:length(physical.cars)
        particles[i] = Vector{Float64}(up.nb_sims)
        weights[i] = ones(up.nb_sims)
        for j in 1:up.nb_sims
            particles[i][j] = aggressiveness(gen, rand(up.rng, gen))
        end
    end
    return AggressivenessBelief(gen, physical, particles, weights)
end


initialize_belief(up::AggressivenessUpdater, d::AggressivenessBelief) = d
