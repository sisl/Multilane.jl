function MDP_fixture()
    dmodel = NoCrashIDMMOBILModel(3, PhysicalParam(3, lane_length=100.0))
    rmodel = NoCrashRewardModel()
    return NoCrashMDP(dmodel, rmodel, 1.0)
end
function state_fixture()
    # XXX are we going to git rid of v_min, etc?
    lane_length=100.0
    pp = PhysicalParam(3, lane_length=lane_length)
    return MLState(false,0.0,0.0, CarState[CarState(lane_length/2.0, 2.0, pp.v_med, 0.0, Multilane.NORMAL,0),
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
#     s = MLState(false,0.0,0.0, CarState[CarState(0.0, 2.0, pp.v_med, 0.0, Multilane.NORMAL),
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
    deleteat!(s.cars, 2)
    rng = MersenneTwister(2)
    sp, r = generate_sr(mdp, s, MLAction(0.0,0.0), rng)
    @test length(sp.cars) == 3
end

"""
Return a CarState for the ego vehicle with the specified y.
"""
ego_state_spec_y(y) = CarState(0.0, y, 30.0, 0.0, Multilane.NORMAL,1)

function test_snapback()
    println("\t\tTesting snapping back after lane change too far")
    mdp = MDP_fixture()
    rng = MersenneTwister(2)
    mdp.dmodel.phys_param.dt = 0.75
    s = state_fixture()
    s.cars[1] = ego_state_spec_y(2.0)
    a1 = MLAction(0,1.0)
    sp,r = generate_sr(mdp, s, a1, rng)
    @test sp.cars[1].y == 2.0 + 1.0*0.75
    a2 = MLAction(0,2.0)
    sp,r = generate_sr(mdp, s, a2, rng)
    @test sp.cars[1].y == 3.0
    s.cars[1] = ego_state_spec_y(1.99)
    a3 = MLAction(0,1.0)
    sp,r = generate_sr(mdp, s, a3, rng)
    @test sp.cars[1].y == 2.0
    s.cars[1] = ego_state_spec_y(2.01)
    a4 = MLAction(0,-1.0)
    sp,r = generate_sr(mdp, s, a4, rng)
    @test sp.cars[1].y == 2.0
    s.cars[1] = ego_state_spec_y(1.01)
    a5 = MLAction(0.,0.1)
    sp,r = generate_sr(mdp, s, a5, rng)
    @test sp.cars[1].y == 1.01 + 0.1*0.75
end

function test_model()
    println("\tTesting Model...")
    test_occupation()
    test_encounter()
    test_snapback()
end
