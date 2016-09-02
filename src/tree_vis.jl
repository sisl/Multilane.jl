function MCTS.node_tag(s::MLState)
    if s.crashed
        return "CRASH"
    else
        desc = "("
        for c in s.cars
            desc = string(desc, @sprintf("[%.1f,%.1f]",c.x,c.y))
        end
        return string(desc,")")
    end
end

function MCTS.tooltip_tag(s::MLState)
    if s.crashed
        return "CRASH"
    else
        desc = "( "
        for c in s.cars
            desc = string(desc, 
            @sprintf("[%.1f,%.1f,v:%.1f,l:%.1f] ", c.x, c.y, c.vel, c.lane_change))
        end
        return string(desc,")")
    end
end

function MCTS.node_tag(a::MLAction)
    return @sprintf("[%.1f,%.1f]", a.acc, a.lane_change)
end

function MCTS.tooltip_tag(a::MLAction)
    return @sprintf("[a:%.1f,l:%.1f]", a.acc, a.lane_change)
end
