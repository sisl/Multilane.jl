function MCTS.node_tag(s::MLState)
    if isnull(s.terminal)
        desc = ""
    else
        desc = "[$(uppercase(string(get(s.terminal))))] "
    end
    desc *= "("
    for c in s.cars
        desc *= @sprintf("[%.1f,%.1f]",c.x,c.y)
    end
    return string(desc,")")
end

function MCTS.tooltip_tag(s::MLState)
    if isnull(s.terminal)
        desc = ""
    else
        desc = "[$(uppercase(string(get(s.terminal))))] "
    end
    desc *= "("
    for c in s.cars
        desc *= @sprintf("[%.1f,%.1f,v:%.1f,l:%.1f] ", c.x, c.y, c.vel, c.lane_change)
    end
    desc *= ")"
    return desc
end

function MCTS.node_tag(a::MLAction)
    return @sprintf("[%.1f,%.1f]", a.acc, a.lane_change)
end

function MCTS.tooltip_tag(a::MLAction)
    return @sprintf("[a:%.1f,l:%.1f]", a.acc, a.lane_change)
end
