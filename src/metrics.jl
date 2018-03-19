abstract type SimMetric end # functions: key(m), datatype(m), calculate(m, mdp, sim)

mutable struct MaxBrakeMetric <: SimMetric end
key(::MaxBrakeMetric) = :max_brake
datatype(::MaxBrakeMetric) = Float64
function calculate(m::MaxBrakeMetric, problem::NoCrashProblem, sim::HistoryRecorder)
    sh = sim.state_hist
    max_brake = 0.0
    for i in 1:length(sh)-1
        mbi = max_braking(problem, sh[i], sh[i+1])
        if mbi > max_brake
            max_brake = mbi
        end
    end
    return max_brake
end

mutable struct NumBehaviorBrakesMetric
    idx::Int
    name::AbstractString
    threshold::Float64
end
key(m::NumBehaviorBrakesMetric) = Symbol(string(m.name, "_nb_brakes"))
datatype(::NumBehaviorBrakesMetric) = Int
function calculate(m::NumBehaviorBrakesMetric, problem::NoCrashProblem, sim::HistoryRecorder)
    sh = sim.state_hist
    nb_brakes = 0
    for i in 1:length(sh)-1
        nb_leaving = 0 
        dt = problem.dmodel.phys_param.dt
        s = sh[i]
        sp = sh[i+1]
        for (i,c) in enumerate(s.cars)
            if length(sp.cars) >= i-nb_leaving
                cp = sp.cars[i-nb_leaving]
            else
                break
            end
            if cp.id != c.id
                nb_leaving += 1
                continue
            else
                if c.behavior.idx == m.idx
                    acc = (cp.vel-c.vel)/dt
                    if acc < -m.threshold 
                        nb_brakes+=1
                    end
                end
            end
        end
    end
    return nb_brakes
end
