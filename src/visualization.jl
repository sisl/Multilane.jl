# using Reel
using AutomotiveDrivingModels
using AutoViz
# using Multilane
# using Interact
# using POMDPToolbox
# using Cairo

function interp_state(s, sp, frac)
    a = frac
    b = 1.0 - frac
    cars = CarState[]
    nb_leaving = 0
    for (i,c) in enumerate(s.cars)
        if length(sp.cars) >= i-nb_leaving
            cp = sp.cars[i-nb_leaving]
        else
            break
        end
        if cp.id != c.id
            nb_leaving += 1
            continue
        else # c and cp have same id
            x = a*cp.x + b*c.x
            y = a*cp.y + b*c.y
            vel = a*cp.vel + b*c.vel
            lane_change = c.lane_change
            id = c.id
            push!(cars, CarState(x, y, vel, lane_change, c.behavior, id))
        end
    end
    x = a*sp.x + b*s.x
    t = a*sp.t + b*s.t
    return MLState(x, t, cars, s.terminal)
end

function visualize(p, s, r;
                   tree=nothing,
                   surface=CairoRGBSurface(AutoViz.DEFAULT_CANVAS_WIDTH, AutoViz.DEFAULT_CANVAS_HEIGHT)
                  )
    pp = p.dmodel.phys_param
    stuff = []
    roadway = gen_straight_roadway(pp.nb_lanes, p.dmodel.max_dist+200.0, lane_width=pp.w_lane)
    push!(stuff, roadway)
    str = @sprintf("t: %6.2f\nx: %6.2f\nvel: %6.2f", s.t, s.x, s.cars[1].vel)
    if r != nothing
        str *= @sprintf("\nr: %6.2f", r)
    end
    push!(stuff, str)
    if tree != nothing
        # push!(stuff, RelativeRender(tree, s.t, s.cars[1].vel))
        # use the velocity at the root so it doesn't move
        v = tree.node.tree.root_belief.physical.cars[1].vel
        push!(stuff, RelativeRender(tree, s.t, v))
    end
    for (i,c) in enumerate(s.cars)
        if i == 1
            color = colorant"green"
        else
            agg = aggressiveness(Multilane.STANDARD_CORRELATED, c.behavior)
            color = weighted_color_mean(agg, colorant"red", colorant"blue")
        end
        ac = ArrowCar([c.x+s.x, (c.y-1.0)*pp.w_lane], id=i, color=color)
        push!(stuff, ac)
    end

    render(stuff, cam=CarFollowCamera(1, 8.5), surface=surface)
end

# start with just lines

struct RelativeRender{T} <: Renderable
    object::T
    t::Float64     # time that this is being rendered 
    vel::Float64
end

struct NodeWithRollouts{N<:POWTreeObsNode}
    node::N
    rollouts::Vector
end

function AutoViz.render!(rm::RenderModel, r::RelativeRender{N}) where {N<:NodeWithRollouts}
    node = r.object.node
    tree = node.tree
    if isroot(node)
        b = tree.root_belief
    else
        b = tree.sr_beliefs[node.node].b
    end
    m = last(tree.sr_beliefs).model
    pp = m.dmodel.phys_param
    t = r.t
    dt = pp.dt

    # draw dots and find o_children
    a_children = tree.tried[node.node]
    o_children = Int[]
    for ac in a_children
        onodes = [last(oi) for oi in tree.generated[ac]]
        append!(o_children, onodes)
        value = tree.v[ac]
        if value > 0.0
            color = weighted_color_mean(value, colorant"yellow", colorant"green")
        else
            color = weighted_color_mean(clamp(-value, 0.0, 1.0), colorant"yellow", colorant"red")
        end
        if !isempty(onodes)
            ps = tree.o_labels[first(onodes)]
            c = first(ps.cars)
            x = ps.x + c.x - (ps.t-t)*r.vel
            # x, y, r, fill, stroke, linewidth
            add_instruction!(rm, render_circle, (x, (c.y-1.0)*pp.w_lane, 0.2, color, color, 0.0))
        end
    end
    o_children


    # draw all the lines
    # XXX hack: should be most likely instead of rand
    s = rand(Base.GLOBAL_RNG, b)
    for oc in o_children
        sp = rand(Base.GLOBAL_RNG, tree.sr_beliefs[oc].b)
        render_rel_lines!(rm, pp, s, sp, t, r.vel)
    end

    # render the children
    for oc in o_children
        nwr = NodeWithRollouts(POWTreeObsNode(tree, oc), r.object.rollouts)
        render!(rm, RelativeRender(nwr, t, r.vel))
        ro = r.object.rollouts[oc]
        for (s, sp) in eachstep(ro, "s,sp")
            render_rel_lines!(rm, pp, s, sp, t, r.vel)
            # render_rel_ego_line!(rm, pp, s, sp, t, r.vel)
        end
    end
end

function render_rel_ego_line!(rm::RenderModel, pp, s, sp, t, vel)
    c = first(s.cars)
    cp = first(sp.cars)
    x1 = s.x + c.x - (s.t-t)*vel 
    y1 = (c.y - 1.0)*pp.w_lane
    x2 = sp.x + cp.x - (sp.t-t)*vel
    y2 = (cp.y - 1.0)*pp.w_lane
    width = 0.1
    color = RGBA(0.8, 0.8, 0.8, 0.2)
    add_instruction!(rm, render_line_segment, (x1, y1, x2, y2, color, width))
end

function render_rel_lines!(rm::RenderModel, pp, s, sp, t, vel)
    nb_leaving = 0
    for (i,c) in enumerate(s.cars)
        if length(sp.cars) >= i-nb_leaving
            cp = sp.cars[i-nb_leaving]
        else
            break
        end
        if cp.id != c.id
            nb_leaving += 1
            continue
        else # c and cp have same id
            # for x, if the point is in the future, you want it to be way behind where it should be
            # if the point is in the past, you want it to be way ahead
            x1 = s.x + c.x - (s.t-t)*vel 
            y1 = (c.y - 1.0)*pp.w_lane
            x2 = sp.x + cp.x - (sp.t-t)*vel
            y2 = (cp.y - 1.0)*pp.w_lane
            width = 0.1
            if c.id == 1
                color = RGBA(0.8, 0.8, 0.8, 0.2)
            else
                agg = aggressiveness(Multilane.STANDARD_CORRELATED, c.behavior)
                color = weighted_color_mean(agg, RGBA(1.0, 0.0, 0.0, 0.2), RGBA(0.0, 0.0, 1.0, 0.2))
            end
            add_instruction!(rm, render_line_segment, (x1, y1, x2, y2, color, width))
        end
    end
end

function make_rollouts(planner, tree)
    t0 = tree.root_belief.physical.t
    m = planner.problem
    dt = m.dmodel.phys_param.dt
    rop = solve(SimpleSolver(), m)
    rollouts = Vector{Any}(length(tree.sr_beliefs))
    for i in 2:length(tree.sr_beliefs)
        srb = tree.sr_beliefs[i]
        b = srb.b
        ps = b.physical
        start_t = srb.b.physical.t
        completed = round(Int, (start_t - t0)/dt)
        steps = planner.solver.max_depth - completed
        hr = HistoryRecorder(max_steps=steps)
        cars = [CarState(ps.cars[i], first(b.particles[i])) for i in 1:length(b.particles)]
        state = MLState(ps, cars)
        mdp = NoCrashMDP{typeof(m.rmodel), typeof(m.dmodel.behaviors)}(m.dmodel, m.rmodel, m.discount, m.throw)
        hist = simulate(hr, mdp, rop, state)
        rollouts[i] = hist
    end
    return rollouts
end

#=
function AutoViz.render!(rm::RenderModel, t::RelativeRender{P}) where {P<:POMCPOWTree}
    render!(rm, RelativeRender(POWTreeObsNode(t.object, 1), t.t, t.vel))
end

function AutoViz.render!(rm::RenderModel, r::RelativeRender{N}) where {N<:POWTreeObsNode}
    node = r.object
    tree = node.tree
    if isroot(node)
        b = tree.root_belief
    else
        b = tree.sr_beliefs[node.node].b
    end
    m = last(tree.sr_beliefs).model
    pp = m.dmodel.phys_param
    t = r.t
    dt = pp.dt

    a_children = tree.tried[node.node]
    # value = maximum(tree.v[c] for c in a_children)
    o_children = Int[]
    for ac in a_children
        append!(o_children, [last(oi) for oi in tree.generated[ac]])
    end
    o_children

    # draw all the lines
    s = b.physical
    for oc in o_children
        sp = tree.sr_beliefs[oc].b.physical
        render_rel_lines!(rm, pp, s, sp, t, r.vel)
    end

    # render the children
    for oc in o_children
        render!(rm, RelativeRender(POWTreeObsNode(tree, oc), t, r.vel))
    end
end

=#


#=
BEHAVIOR_COLORS = Dict{Float64,AbstractString}(0.5=>"#0101DF",0.25=>"#D7DF01",0.0=>"#FF0000")

function visualize(mdp::Union{MLMDP,MLPOMDP}, s::MLState, a::MLAction, sp::MLState;
                   idx::Nullable{Int}=Nullable{Int}())
    pp = mdp.dmodel.phys_param
    roadway = gen_straight_roadway(pp.nb_lanes,
                                   pp.lane_length,
                                   lane_width=pp.w_lane)

    hbol = HardBrakeOverlay(pp, braking_ids(mdp, s, sp))
    iol = InfoOverlay(pp, idx,
                      s.cars[1].vel,
                      max_braking(mdp, s, sp),
                      is_crash(mdp, s, sp))
    cidol = CarIDOverlay()
    cvol = CarVelOverlay()

    scene = Scene()
    for cs in s.cars
        push!(scene, Vehicle(VehicleState(VecSE2(cs.x, (cs.y-1.0)*pp.w_lane, 0.0), roadway, cs.vel), 
                                VehicleDef(cs.id, AgentClass.CAR, pp.l_car, pp.w_car)))
    end
    render(scene, roadway, [hbol, iol, cidol, cvol], cam=FitToContentCamera())
end

visualize(mdp::Union{MLMDP,MLPOMDP}, s::MLState) = visualize(mdp.dmodel.phys_param, s)

function visualize(pp::PhysicalParam, s::MLState)
    roadway = gen_straight_roadway(pp.nb_lanes,
                                   pp.lane_length,
                                   lane_width=pp.w_lane)
    scene = Scene()
    for cs in s.cars
        push!(scene, Vehicle(VehicleState(VecSE2(cs.x, (cs.y-1.0)*pp.w_lane, 0.0), roadway, cs.vel), 
                                VehicleDef(cs.id, AgentClass.CAR, pp.l_car, pp.w_car)))
    end
    render(scene, roadway, [CarIDOverlay(), CarVelOverlay()], cam=FitToContentCamera())
end

mutable struct HardBrakeOverlay <: SceneOverlay
    pp::PhysicalParam
    ids::Vector{Int}
end

function AutoViz.render!(rm::RenderModel, o::HardBrakeOverlay, scene::Scene, roadway::Roadway)
    for veh in scene
        if veh.def.id in o.ids
            for (size, offset) in [(1.2,0.4), (1.0,0.8), (0.8,1.2)]
                top = VecE2(veh.state.posG) + VecE2(-((1.+offset)*o.pp.l_car/2), size*o.pp.w_car/2)
                bottom = VecE2(veh.state.posG) + VecE2(-((1.+offset)*o.pp.l_car/2), -size*o.pp.w_car/2)
                add_instruction!(rm, render_line_segment, (top.x, top.y, bottom.x, bottom.y, colorant"red", 0.3))
            end
        end
    end
    # if !isempty(o.ids)
    #     add_instruction!(rendermodel, render_text, ("Hard Brake(s)!", 0, 15, 12, colorant"red"))
    # end
end

mutable struct InfoOverlay <: SceneOverlay
    pp::PhysicalParam
    state_index::Nullable{Int}
    vel::Float64
    max_braking::Float64
    iscrash::Bool
end

function AutoViz.render!(rm::RenderModel, o::InfoOverlay, scene::Scene, roadway::Roadway)
    line_delta = 1.6
    top_line_y = -o.pp.w_lane/2.0 - line_delta
    y = top_line_y
    if !isnull(o.state_index)
        add_instruction!(rm, render_text,
                         ("state index: $(get(o.state_index))", 0, y, 12, colorant"white"))
        y -= line_delta
    end
    add_instruction!(rm, render_text,
                     (@sprintf("velocity: %5.2f m/s", o.vel), 0, y, 12, colorant"white"))
    y -= line_delta
    add_instruction!(rm, render_text,
                     (@sprintf("max braking: %5.2f m/s", o.max_braking), 0, y, 12,
                      o.max_braking==o.pp.brake_limit ? colorant"red" : colorant"white"))
    y -= line_delta
    if o.iscrash
        add_instruction!(rm, render_text,
                         ("crash!", 0, y, 12, colorant"red"))
        y -= line_delta
    end
end

mutable struct CarIDOverlay <: SceneOverlay end

function AutoViz.render!(rm::RenderModel, o::CarIDOverlay, scene::Scene, roadway::Roadway)
    for (i,v) in enumerate(scene)
        cx = v.state.posG.x
        cy = v.state.posG.y
        nx = cx - v.def.length/2
        ny = cy + v.def.width/2 - 0.6
        add_instruction!(rm, render_text,
                         (@sprintf("%02.d",i), nx, ny, 7, colorant"white"))
        idx = cx - v.def.length/2
        idy = cy - v.def.width/2 + 0.1
        add_instruction!(rm, render_text,
                         (@sprintf("id %02.d",v.def.id), idx, idy, 7, colorant"white"))
    end
end

mutable struct CarVelOverlay <: SceneOverlay end

function AutoViz.render!(rm::RenderModel, o::CarVelOverlay, scene::Scene, roadway::Roadway)
    for (i,v) in enumerate(scene)
        cx = v.state.posG.x
        cy = v.state.posG.y
        vx = cx + v.def.length/2 - 1.4
        vy = cy + v.def.width/2 - 0.6
        add_instruction!(rm, render_text,
                         (@sprintf("%04.1f",v.state.v), vx, vy, 7, colorant"white"))
    end
end
=#
