function MDP_fixture()
    dmodel = NoCrashIDMMOBILModel(3, PhysicalParam(3, lane_length=100.0))
    rmodel = NoCrashRewardModel()
    return NoCrashMDP(dmodel, rmodel, 1.0)
end
function state_fixture()
    # XXX are we going to git rid of v_min, etc?
    lane_length=100.0
    pp = PhysicalParam(3, lane_length=lane_length)
    return MLState(false, CarState[CarState(lane_length/2.0, 2.0, pp.v_med, 0.0, Nullable{BehaviorModel}(),0),
                                CarState(lane_length*3/4, 1.0, pp.v_min, 0.0,
                                         IDMMOBILBehavior("normal", pp.v_min, 0.0, 1),0),
                                CarState(lane_length*1/4, 3.0, pp.v_max, 0.0,
                                         IDMMOBILBehavior("normal", pp.v_max, 0.0, 2),0)])
end

function test_occupation()
    println("\t\tTestng occupation_overlap")
    @test Multilane.occupation_overlap(1.0, 2.0) == false
    @test Multilane.occupation_overlap(1.0, 1.99) == true
    @test Multilane.occupation_overlap(0.9, 2.0) == false
    @test Multilane.occupation_overlap(0.9, 1.1) == true
    @test Multilane.occupation_overlap(0.9, 1.99) == true
end

# not done
# function test_e_brake()
#     mdp = MDP_fixture()
#     pp = mdp.dmodel.phys_param
#     s = MLState(false, CarState[CarState(0.0, 2.0, pp.v_med, 0.0, Nullable{BehaviorModel}()),
#                                 CarState(pp.l_car+0.1, 2.0, pp.v_min, 0.0,
#                                          BehaviorModel("normal"), pp.v_min, 0.0, 1)])
# end

"""
Test that cars are regenerated when there are some missing
"""
function test_encounter()
    println("\t\tTesting car regeneration for encounters")
    mdp = MDP_fixture()
    mdp.dmodel.p_appear = 1.0
    s = state_fixture()
    deleteat!(s.env_cars, 2)
    rng = MersenneTwister(2)
    sp, r = generate_sr(mdp, s, MLAction(0.0,0.0), rng)
    @test length(sp.env_cars) == 3
end

function test_model()
    println("\tTesting Model...")
    test_occupation()
    test_encounter()
end
