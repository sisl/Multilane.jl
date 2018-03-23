function relaxed_initial_state(mdp::NoCrashProblem, steps=200,
                             rng=MersenneTwister(rand(UInt32)); 
                             solver=BehaviorSolver(NORMAL, true, rng))
    
    mdp = deepcopy(mdp)
    mdp.dmodel.max_dist = Inf
    mdp.dmodel.brake_terminate_thresh = Inf
    mdp.dmodel.lane_terminate = false
    pp = mdp.dmodel.phys_param
    is = MLState(0.0, 0.0, CarState[CarState(pp.lane_length/2, 1.0, pp.v_med, 0.0, NORMAL, 1)])
    sim = HistoryRecorder(max_steps=steps, rng=rng)
    policy = solve(solver, mdp)
    hist = simulate(sim, mdp, policy, is)
    s = last(state_hist(hist))
    s.t = 0.0
    s.x = 0.0
    @assert s.cars[1].y == 1.0
    @assert isnull(s.terminal)
    return s
end
