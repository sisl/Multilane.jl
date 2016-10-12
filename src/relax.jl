function relaxed_initial_state(mdp::NoCrashProblem, steps=200,
                             rng=MersenneTwister(rand(UInt32)); 
                             solver=BehaviorSolver(NORMAL, true, rng))
    pp = mdp.dmodel.phys_param
    is = MLState(false, false, 0.0, 0.0, CarState[CarState(pp.lane_length/2, 1.0, pp.v_med, 0.0, NORMAL, 1)])
    sim = HistoryRecorder(max_steps=steps, rng=rng)
    policy = solve(solver, mdp)
    simulate(sim, mdp, policy, is)
    return sim.state_hist[end]
end
