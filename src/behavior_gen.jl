abstract BehaviorGenerator

type DiscreteBehaviorSet <: BehaviorGenerator
    models::Vector{BehaviorModel}
    weights::WeightVec
end

rand(rng, s::DiscreteBehaviorSet) = sample(rng, s.models, s.weights)
