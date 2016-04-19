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

function test_car_neighborhood_creation()
	println("\t\tTesting Car Neighborhood Creation")
	p1 = CarNeighborhood()

end

function test_car_neighborhood_equality()
	println("\t\tTesting Car Neighborhood Equality")
	#Nothing to do--such functions weren't created
end

function test_car_neighborhood_hashing()
	println("\t\tTesting Car Neighborhood Hashing")
	#nothing to do--such functions weren't created
end

function test_get_adj_cars()
	println("\t\tTesting get_adj_cars")
	##TODO: assertions
	nb_lanes = 3
	pp = PhysicalParam(nb_lanes,nb_vel_bins=5,lane_length=12.)
	bs = IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in enumerate(product(["cautious","normal","aggressive"],[pp.v_slow;pp.v_med;pp.v_fast],[pp.l_car]))]
	#CASE: just agent car
	cs = CarState[CarState((6.,3,),31.,0,bs[1])]
	dp1 = 6.-pp.l_car
	dp2 = 6.-pp.l_car
	#just make sure it doesn't explode--this should be handled by get_mobil_lane_change
	nbhd = get_adj_cars(pp,cs,1)
	assert(get(nbhd.ahead_dist,0,-1)==-1)
	assert(get(nbhd.ahead_dist,1,-1)==-1)
	assert(get(nbhd.ahead_dist,-1,-1)==-1)
	assert(get(nbhd.behind_dist,0,-1)==-1)
	assert(get(nbhd.behind_dist,1,-1)==-1)
	assert(get(nbhd.behind_dist,-1,-1)==-1)
	#CASE: nobody to the left
	cs = CarState[CarState((12.,3,),31.,0,bs[1]),CarState((6.,3,),31.,0,bs[1]),CarState((0.,3,),31.,0,bs[1]),CarState((12.,1,),31.,0,bs[1]),CarState((0.,1,),31.,0,bs[1])]
	nbhd = get_adj_cars(pp,cs,2)
	assert(get(nbhd.ahead_dist,0,-1),dp1)
	assert(get(nbhd.ahead_dist,1,-1),-1)
	assert(get(nbhd.ahead_dist,-1,-1),dp1)
	assert(get(nbhd.behind_dist,0,-1),dp2)
	assert(get(nbhd.behind_dist,1,-1),-1)
	assert(get(nbhd.behind_dist,-1,-1),dp2)
	#CASE: nobody to the right
	cs = CarState[CarState((12.,3,),31.,0,bs[1]),CarState((0.,3,),31.,0,bs[1]),CarState((6.,3,),31.,0,bs[1]),CarState((12.,5,),31.,0,bs[1]),CarState((0.,5,),31.,0,bs[1])]
	nbhd = get_adj_cars(pp,cs,3)
	assert(get(nbhd.ahead_dist,0,-1),dp1)
	assert(get(nbhd.ahead_dist,1,-1),dp1)
	assert(get(nbhd.ahead_dist,-1,-1),-1)
	assert(get(nbhd.behind_dist,0,-1),dp2)
	assert(get(nbhd.behind_dist,1,-1),dp2)
	assert(get(nbhd.behind_dist,-1,-1),-1)
	#CASE: no one ahead
	cs = CarState[CarState((0.,5,),31.,0,bs[1]),CarState((0.,3,),31.,0,bs[1]),CarState((0.,1,),31.,0,bs[1]),CarState((6.,3,),31.,0,bs[1])]
	nbhd = get_adj_cars(pp,cs,4)
	assert(get(nbhd.ahead_dist,0,-1),-1)
	assert(get(nbhd.ahead_dist,1,-1),-1)
	assert(get(nbhd.ahead_dist,-1,-1),-1)
	assert(get(nbhd.behind_dist,0,-1),dp2)
	assert(get(nbhd.behind_dist,1,-1),dp2)
	assert(get(nbhd.behind_dist,-1,-1),dp2)
	#CASE: no one behind
	cs = CarState[CarState((12.,5,),31.,0,bs[1]),CarState((12.,3,),31.,0,bs[1]),CarState((12.,1,),31.,0,bs[1]),CarState((6.,3,),31.,0,bs[1])]
	nbhd = get_adj_cars(pp,cs,4)
	assert(get(nbhd.ahead_dist,0,-1),dp1)
	assert(get(nbhd.ahead_dist,1,-1),dp1)
	assert(get(nbhd.ahead_dist,-1,-1),dp1)
	assert(get(nbhd.behind_dist,0,-1),-1)
	assert(get(nbhd.behind_dist,1,-1),-1)
	assert(get(nbhd.behind_dist,-1,-1),-1)
	#CASE: full house
	cs = CarState[CarState((12.,3,),31.,0,bs[1]),CarState((0.,3,),31.,0,bs[1]),CarState((12.,1,),31.,0,bs[1]),CarState((0.,1,),31.,0,bs[1]),CarState((12.,5,),35.,0,bs[1]),CarState((0.,5,),35.,0,bs[1]),CarState((6.,3,),31.,0,bs[1])]
	nbhd = get_adj_cars(pp,cs,7)
	assert(get(nbhd.ahead_dist,0,-1),dp1)
	assert(get(nbhd.ahead_dist,1,-1),dp1)
	assert(get(nbhd.ahead_dist,-1,-1),dp1)
	assert(get(nbhd.behind_dist,0,-1),dp2)
	assert(get(nbhd.behind_dist,1,-1),dp2)
	assert(get(nbhd.behind_dist,-1,-1),dp2)

	##TODO, someday: test cases for slightly off line cars, cars that are far away
end

function test_get_mobil_lane_change()
	println("\t\tTesting get_mobil_lane_change")
	nb_lanes = 2
	pp = PhysicalParam(nb_lanes,nb_vel_bins=5,lane_length=12.)
	bs = IDMMOBILBehavior[IDMMOBILBehavior(x[1],x[2],x[3],idx) for (idx,x) in enumerate(product(["cautious","normal","aggressive"],[pp.v_slow;pp.v_med;pp.v_fast],[pp.l_car]))]
	#CASE: it's faster, but there's no space--is this even a real case?
	#CASE: it's faster and there is space
	cs = CarState[CarState((12.,3,),35.,0,IDMMOBILBehavior("aggressive",35.,4.,1)),CarState((11.,1,),27.,0,IDMMOBILBehavior("cautious",27.,4.,1)),CarState((6.,1,),31.,0,IDMMOBILBehavior("aggressive",35.,4.,1))]
	nbhd = get_adj_cars(pp,cs,3)
	assert(get_mobil_lane_change(pp,cs[3],nbhd),1)
	#CASE: it's slower and there is space
	cs = CarState[CarState((12.,3,),31.,0,IDMMOBILBehavior("cautious",31.,4.,1)),CarState((11.,1,),35.,0,IDMMOBILBehavior("cautious",35.,4.,1)),CarState((6.,1,),35.,0,IDMMOBILBehavior("aggressive",35.,4.,1))]
	nbhd = get_adj_cars(pp,cs,3)
	assert(get_mobil_lane_change(pp,cs[3],nbhd),0)
	#CASE: someone is going fast behind me and i'm slow
	cs = CarState[CarState((0.,1,),35.,0,IDMMOBILBehavior("aggressive",35.,4.,1)),CarState((6.,1,),27.,0,IDMMOBILBehavior("cautious",27.,4.,1))]
	nbhd = get_adj_cars(pp,cs,2)
	assert(get_mobil_lane_change(pp,cs[2],nbhd),1)
	###repeat for other side
	#CASE: it's faster and there is space
	cs = CarState[CarState((12.,1,),35.,0,IDMMOBILBehavior("aggressive",35.,4.,1)),CarState((11.,3,),27.,0,IDMMOBILBehavior("cautious",27.,4.,1)),CarState((6.,3,),31.,0,IDMMOBILBehavior("aggressive",35.,4.,1))]
	nbhd = get_adj_cars(pp,cs,3)
	assert(get_mobil_lane_change(pp,cs[3],nbhd),-1)
	#CASE: it's slower and there is space
	cs = CarState[CarState((12.,1,),31.,0,IDMMOBILBehavior("cautious",31.,4.,1)),CarState((11.,3,),35.,0,IDMMOBILBehavior("cautious",35.,4.,1)),CarState((6.,3,),35.,0,IDMMOBILBehavior("aggressive",35.,4.,1))]
	nbhd = get_adj_cars(pp,cs,3)
	assert(get_mobil_lane_change(pp,cs[3],nbhd),0)
	#CASE: someone is going fast behind me and i'm slow
	cs = CarState[CarState((0.,3,),35.,0,IDMMOBILBehavior("aggressive",35.,4.,1)),CarState((6.,3,),27.,0,IDMMOBILBehavior("cautious",27.,4.,1))]
	nbhd = get_adj_cars(pp,cs,2)
	assert(get_mobil_lane_change(pp,cs[2],nbhd),-1)
end

function test_mobil()
	println("\tTesting MOBIL Units...")
	test_mobil_creation()
	test_mobil_equality()
	test_mobil_hashing()
	test_car_neighborhood_creation()
	test_car_neighborhood_equality()
	test_car_neighborhood_hashing()
	test_get_adj_cars()
	test_get_mobil_lane_change()
	#NOTE: not working, but since we'll probably be limited to 1 car in the foreseeable future, this isn't the highest priority thing
end
