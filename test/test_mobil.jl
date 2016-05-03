function test_mobil_creation()
	println("\t\tTesting MOBIL Creation")
	p1 = MOBILParam()
	p2 = MOBILParam(p=0.5)
	p3 = MOBILParam(b_safe=5.,a_thr=0.3)
	ps = [MOBILParam(s) for s in ["cautious";"normal";"aggressive"]]
end

function test_mobil_equality()
	#println("\t\tTesting MOBIL Equality")
	p1 = MOBILParam()
	p2 = MOBILParam(p=0.5)
	p3 = MOBILParam(b_safe=5.,a_thr=0.3)
	ps = [MOBILParam(s) for s in ["cautious";"normal";"aggressive"]]
	test_equality("MOBIL",ps)
	assert(p1 == p1)
	assert(p1 != p2)
	assert(p2 != p1)
	assert(p1 == MOBILParam(0.25,4.,0.2))
	assert(MOBILParam(0.25,4.,0.2) == p1)
end

function test_mobil_hashing()
	println("\t\tTesting MOBIL Hashing")
	p1 = MOBILParam()
	p2 = MOBILParam(p=0.5)
	p3 = MOBILParam(b_safe=5.,a_thr=0.3)
	ps = [MOBILParam(s) for s in ["cautious";"normal";"aggressive"]]
	test_hashing("MOBIL",ps)
end

function test_get_neighborhood()
	println("\t\tTesting get_neighborhood")
	##TODO: assertions
	nb_lanes = 3
	pp = PhysicalParam(nb_lanes,nb_vel_bins=5,lane_length=12.)
	bs = IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in enumerate(product(["cautious","normal","aggressive"],[pp.v_slow;pp.v_med;pp.v_fast],[pp.l_car]))]
	#CASE: just agent car
	cs = CarState[CarState(6.,3,31.,0,bs[1])]
	dp1 = 6.-pp.l_car
	dp2 = 6.-pp.l_car
	_a = MLAction(0,0)

	#just make sure it doesn't explode--this should be handled by get_mobil_lane_change
	dmodel = IDMMOBILModel(length(cs), pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(dmodel,s,1)
	assert(length(nbhd) == 6)
	for jdx in nbhd
		assert(jdx == 0)
	end
	#CASE: nobody to the left
	cs = CarState[CarState(12.,3,31.,0,bs[1]),CarState(6.,3,31.,0,bs[1]),CarState(0.,3,31.,0,bs[1]),CarState(12.,1,31.,0,bs[1]),CarState(0.,1,31.,0,bs[1])]

	dmodel = IDMMOBILModel(length(cs), pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(dmodel,s,2)
	assert(length(nbhd) == 6)
	for jdx in nbhd[[3;6]]
		assert(jdx == 0)
	end
	assert(nbhd[1] , 4)
	assert(nbhd[2] , 1)
	assert(nbhd[4] , 5)
	assert(nbhd[5] , 3)
	#CASE: nobody to the right
	cs = CarState[CarState(12.,3,31.,0,bs[1]),CarState(0.,3,31.,0,bs[1]),CarState(6.,3,31.,0,bs[1]),CarState(12.,5,31.,0,bs[1]),CarState(0.,5,31.,0,bs[1])]

	dmodel = IDMMOBILModel(length(cs), pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(dmodel,s,3)
	assert(length(nbhd) == 6)
	for jdx in nbhd[[1;4]]
		assert(jdx , 0)
	end
	assert(nbhd[3] , 4)
	assert(nbhd[2] , 1)
	assert(nbhd[6] , 5)
	assert(nbhd[5] , 2)
	#CASE: no one ahead
	cs = CarState[CarState(0.,5,31.,0,bs[1]),CarState(0.,3,31.,0,bs[1]),CarState(0.,1,31.,0,bs[1]),CarState(6.,3,31.,0,bs[1])]

	dmodel = IDMMOBILModel(length(cs), pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(dmodel,s,4)
	assert(length(nbhd) , 6)
	for jdx in nbhd[[1;2;3]]
		assert(jdx , 0)
	end
	assert(nbhd[4] , 3)
	assert(nbhd[5] , 2)
	assert(nbhd[6] , 1)
	#CASE: no one behind
	cs = CarState[CarState(12.,5,31.,0,bs[1]),CarState(12.,3,31.,0,bs[1]),CarState(12.,1,31.,0,bs[1]),CarState(6.,3,31.,0,bs[1])]

	dmodel = IDMMOBILModel(length(cs), pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(dmodel,s,4)
	assert(length(nbhd) , 6)
	for jdx in nbhd[[4;5;6]]
		assert(jdx , 0)
	end
	assert(nbhd[1] , 3)
	assert(nbhd[2] , 2)
	assert(nbhd[3] , 1)
	#CASE: full house
	cs = CarState[CarState(12.,3,31.,0,bs[1]),CarState(0.,3,31.,0,bs[1]),CarState(12.,1,31.,0,bs[1]),CarState(0.,1,31.,0,bs[1]),CarState(12.,5,35.,0,bs[1]),CarState(0.,5,35.,0,bs[1]),CarState(6.,3,31.,0,bs[1])]

	dmodel = IDMMOBILModel(length(cs), pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(dmodel,s,7)
	assert(length(nbhd) , 6)
	for jdx in nbhd
		assert(jdx != 0)
	end
	assert(nbhd[1] , 3)
	assert(nbhd[2] , 1)
	assert(nbhd[3] , 5)
	assert(nbhd[4] , 4)
	assert(nbhd[5] , 2)
	assert(nbhd[6] , 6)
end
#TODO test_get_dv_ds(); FIX VVV

function test_is_lanechange_dangerous()
	println("\t\tTesting is_lanechange_dangerous")

	nb_lanes = 3
	_a = MLAction(0,0)
	pp = PhysicalParam(nb_lanes,nb_vel_bins=5,lane_length=12.)
	bs = IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in enumerate(product(["cautious","normal","aggressive"],[pp.v_slow;pp.v_med;pp.v_fast],[pp.l_car]))]
	#CASE: no one, in the way
	cs = CarState[CarState(6.,3,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1))]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(dmodel,s,1)
	assert(is_lanechange_dangerous(dmodel,s,nbhd,1,0),false)
	assert(is_lanechange_dangerous(dmodel,s,nbhd,1,1),false)
	assert(is_lanechange_dangerous(dmodel,s,nbhd,1,-1),false)

	#CASE: person close by, same velocity
	cs = CarState[CarState(6.,3,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1)),CarState(12.,5,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1))]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(dmodel,s,1)
	assert(is_lanechange_dangerous(dmodel,s,nbhd,1,1),false)

	#slightly behind
	cs = CarState[CarState(6.,3,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1)),CarState(0.,5,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1))]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(dmodel,s,1)
	assert(is_lanechange_dangerous(dmodel,s,nbhd,1,1),false)
	#CASE: person a little behind, slower
	cs = CarState[CarState(6.,3,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1)),CarState(1.,5,32.,0,IDMMOBILBehavior("aggressive",35.,4.,1))]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(dmodel,s,1)
	assert(is_lanechange_dangerous(dmodel,s,nbhd,1,1),false)
	#CASE: person a little ahead, faster
	cs = CarState[CarState(6.,3,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1)),CarState(1.,5,32.,0,IDMMOBILBehavior("aggressive",35.,4.,1))]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(dmodel,s,2)
	assert(is_lanechange_dangerous(dmodel,s,nbhd,2,-1),false)
	#CASE: person a little behind, faster
	cs = CarState[CarState(6.,3,32.,0,IDMMOBILBehavior("aggressive",35.,4.,1)),CarState(1.,5,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1))]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(dmodel,s,1)
	assert(is_lanechange_dangerous(dmodel,s,nbhd,1,1),true)
	#CASE: person a little ahead, slower
	cs = CarState[CarState(6.,3,32.,0,IDMMOBILBehavior("aggressive",35.,4.,1)),CarState(1.,5,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1))]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(dmodel,s,2)
	assert(is_lanechange_dangerous(dmodel,s,nbhd,2,-1),true)
	#CASE: next to each other
	cs = CarState[CarState(6.,3,32.,0,IDMMOBILBehavior("aggressive",35.,4.,1)),CarState(6.,5,33.,0,IDMMOBILBehavior("aggressive",35.,4.,1))]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(dmodel,s,2)
	assert(is_lanechange_dangerous(dmodel,s,nbhd,2,-1),true)
end

function test_get_rear_accel()
	println("\t\tTesting get_rear_accel")
	nb_lanes = 3
	_a = MLAction(0,0)
	pp = PhysicalParam(nb_lanes,nb_vel_bins=5,lane_length=12.)
	bs = IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in enumerate(product(["cautious","normal","aggressive"],[pp.v_slow;pp.v_med;pp.v_fast],[pp.l_car]))]
	#No one in that position
	cs = CarState[CarState(6.,3,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1))]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(dmodel,s,1)
	a,a_ = get_rear_accel(dmodel,s,nbhd,1,0)
	assert(a,0.)
	assert(a_,0.)
	a,a_ = get_rear_accel(dmodel,s,nbhd,1,1)
	assert(a,0.)
	assert(a_,0.)
	a,a_ = get_rear_accel(dmodel,s,nbhd,1,-1)
	assert(a,0.)
	assert(a_,0.)
	#will go to free flow
	cs = CarState[CarState(6.,3,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1)),
								CarState(0.,5,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1)),
								CarState(0.,3,31.,0,IDMMOBILBehavior("aggressive",31.,4.,1)),
								CarState(0.,1,27.,0,IDMMOBILBehavior("aggressive",27.,4.,1))]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(dmodel,s,1)
	a,a_ = get_rear_accel(dmodel,s,nbhd,1,0)
	assert(a_,a,>=)
	a,a_ = get_rear_accel(dmodel,s,nbhd,1,1)
	assert(a_,a,>=)
	a,a_ = get_rear_accel(dmodel,s,nbhd,1,-1)
	assert(a_,a,>=)
	#some synthetic case
end

function test_get_mobil_lane_change()
	println("\t\tTesting get_mobil_lane_change")
	nb_lanes = 2
	_a = MLAction(0,0)
	pp = PhysicalParam(nb_lanes,nb_vel_bins=5,lane_length=12.)
	bs = IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in enumerate(product(["cautious","normal","aggressive"],[pp.v_slow;pp.v_med;pp.v_fast],[pp.l_car]))]
	#CASE: it's faster, but there's no space--is this even a real case?
	#CASE: it's faster and there is space
	cs = CarState[CarState(12.,3,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1)),
								CarState(11.,1,27.,0,IDMMOBILBehavior("cautious",27.,4.,1)),
								CarState(6.,1,31.,0,IDMMOBILBehavior("aggressive",35.,4.,1))]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(dmodel,s,3)
	assert(get_mobil_lane_change(dmodel,s,nbhd,3),1)
	#CASE: it's slower and there is space
	cs = CarState[CarState(12.,3,31.,0,IDMMOBILBehavior("cautious",31.,4.,1)),CarState(11.,1,35.,0,IDMMOBILBehavior("cautious",35.,4.,1)),CarState(6.,1,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1))]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(dmodel,s,3)
	assert(get_mobil_lane_change(dmodel,s,nbhd,3),0)
	#CASE: someone is going fast behind me and i'm slow
	cs = CarState[CarState(0.,1,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1)),CarState(6.,1,27.,0,IDMMOBILBehavior("cautious",27.,4.,1))]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(dmodel,s,2)
	assert(get_mobil_lane_change(dmodel,s,nbhd,2),1)
	###repeat for other side
	#CASE: it's faster and there is space
	cs = CarState[CarState(12.,1,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1)),CarState(11.,3,27.,0,IDMMOBILBehavior("cautious",27.,4.,1)),CarState(6.,3,31.,0,IDMMOBILBehavior("aggressive",35.,4.,1))]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(dmodel,s,3)
	assert(get_mobil_lane_change(dmodel,s,nbhd,3),-1)
	#CASE: it's slower and there is space
	cs = CarState[CarState(12.,1,31.,0,IDMMOBILBehavior("cautious",31.,4.,1)),CarState(11.,3,35.,0,IDMMOBILBehavior("cautious",35.,4.,1)),CarState(6.,3,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1))]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(dmodel,s,3)
	assert(get_mobil_lane_change(dmodel,s,nbhd,3),0)
	#CASE: someone is going fast behind me and i'm slow
	cs = CarState[CarState(0.,3,35.,0,IDMMOBILBehavior("aggressive",35.,4.,1)),CarState(6.,3,27.,0,IDMMOBILBehavior("cautious",27.,4.,1))]
	dmodel = IDMMOBILModel(length(cs),pp)
	s = MLState(false,cs)
	nbhd = get_neighborhood(dmodel,s,2)
	assert(get_mobil_lane_change(dmodel,s,nbhd,2),-1)
end

function test_mobil()
	println("\tTesting MOBIL Units...")
	test_mobil_creation()
	test_mobil_equality()
	test_mobil_hashing()
	test_get_neighborhood()
	test_is_lanechange_dangerous()
	test_get_rear_accel()
	test_get_mobil_lane_change()
	#NOTE: not working, but since we'll probably be limited to 1 car in the foreseeable future, this isn't the highest priority thing
end
