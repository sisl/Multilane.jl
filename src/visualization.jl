# using PyPlot
#=
module MLVisualization

export
    visualize,
    display_sim,
    write_tmp_gif
=#
using Reel
using AutomotiveDrivingModels
using AutoViz
using Multilane
using Interact
using POMDPToolbox
using Cairo

BEHAVIOR_COLORS = Dict{Float64,AbstractString}(0.5=>"#0101DF",0.25=>"#D7DF01",0.0=>"#FF0000")

function display_sim(mdp, S::Array{MLState,1}, A::Array{MLAction,1})
    warn("This should be run in a Jupyter Notebook")
    assert(length(S) == length(A)+1)
    @manipulate for i = 1:length(A)
        visualize(mdp,S[i],A[i],S[i+1])
    end
end

show_state(mdp::Union{MLMDP, MLPOMDP}, s::MLState) = show_state(mdp.dmodel.phys_param, s)

function show_state(pp::PhysicalParam, s::MLState)
    c = visualize(pp, s)
    tn = string(tempname(), ".png")
    write_to_png(c, tn)
    run(`xdg-open $tn`)
end

function show_sim(mdp, sim::HistoryRecorder)
    f = write_tmp_gif(mdp, sim)
    run(`gifview $f`)
end

function display_sim(mdp, sim::HistoryRecorder)
    display_sim(mdp, sim.state_hist, sim.action_hist)
end

function save_frame(mdp, sim, k=1, filename=string(tempname(),".png"))
    c = visualize(mdp, sim.state_hist[k], sim.action_hist[k], sim.state_hist[k+1])
    write_to_png(c, filename)
    println("Saved to $filename")
end

function write_tmp_gif(mdp, sim::HistoryRecorder)
    dt = mdp.dmodel.phys_param.dt
    S = sim.state_hist
    A = sim.action_hist
    length(sim.action_hist)
    film = roll(fps = 1/dt, duration = dt*(length(sim.state_hist))) do t, dt
        i = round(Int, t/dt)+1
        if i == length(sim.state_hist)
            visualize(mdp, S[i])
        else
            visualize(mdp, S[i], A[i], S[i+1], idx=Nullable(i))
        end
    end
    filename = string(tempname(), ".gif")
    write(filename, film)
    return filename
end

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
