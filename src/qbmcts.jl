struct QBSolver <: Solver
    solver
end

solve(sol::QBSolver, p::Union{POMDP,MDP}) = QBPlanner(solve(sol.solver, p))

struct QBPlanner{P<:DPWPlanner} <: Policy
    p::P
end

function POMDPToolbox.action_info(qp::QBPlanner, b; tree_in_info=false)
    p = qp.p
    dpw = p
    sol = p.solver

    local a::POMDPs.action_type(p.mdp)
    info = Dict{Symbol, Any}()
    try
        total_n = 0
        @assert p.solver.enable_action_pw == false
        S = state_type(p.mdp)
        A = POMDPs.action_type(p.mdp)
        tree = MCTS.DPWTree{S,A}(p.solver.n_iterations)
        p.tree = Nullable(tree)

        banodes = Int[]

        for a in iterator(actions(dpw.mdp, b))
            n0 = init_N(sol.init_N, dpw.mdp, b, a)
            anode = insert_action_orphan!(tree, a, n0,
                                          init_Q(sol.init_Q, dpw.mdp, b, a),
                                          false)
            push!(banodes, anode)
            total_n += n0
        end

        i = 0
        start_us = CPUtime_us()
        for i = 1:p.solver.n_iterations
            s = rand(p.rng, b)
            @assert !isterminal(p.mdp, s)

            best_UCB = -Inf
            sanode = 0
            ltn = log(total_n)
            for child in banodes
                n = tree.n[child]
                q = tree.q[child]
                if ltn <= 0 && n == 0
                    UCB = q
                else
                    c = sol.exploration_constant # for clarity
                    UCB = q + c*sqrt(ltn/n)
                end
                @assert !isnan(UCB)
                @assert !isequal(UCB, -Inf)
                if UCB > best_UCB
                    best_UCB = UCB
                    sanode = child
                end
            end

            a = tree.a_labels[sanode]

            new_node = false
            if tree.n_a_children[sanode] <= sol.k_state*tree.n[sanode]^sol.alpha_state
                sp, r = generate_sr(dpw.mdp, s, a, dpw.rng)

                spnode = sol.check_repeat_state ? get(tree.s_lookup, sp, 0) : 0

                if spnode == 0 # there was not a state node for sp already in the tree
                    spnode = MCTS.insert_state_node!(tree, sp, sol.keep_tree || sol.check_repeat_state)
                    new_node = true
                end
                push!(tree.transitions[sanode], (spnode, r))

                if !sol.check_repeat_state 
                    tree.n_a_children[sanode] += 1
                elseif !((sanode,spnode) in tree.unique_transitions)
                    push!(tree.unique_transitions, (sanode,spnode))
                    tree.n_a_children[sanode] += 1
                end
            else
                spnode, r = rand(dpw.rng, tree.transitions[sanode])
            end

            if new_node
                q = r + discount(dpw.mdp)*MCTS.estimate_value(dpw.solved_estimate, dpw.mdp, sp, dpw.solver.depth-1)
            else
                q = r + discount(dpw.mdp)*MCTS.simulate(dpw, spnode, dpw.solver.depth-1)
            end

            tree.n[sanode] += 1
            total_n += 1

            tree.q[sanode] += (q - tree.q[sanode])/tree.n[sanode]

            if CPUtime_us() - start_us >= p.solver.max_time * 1e6
                break
            end
        end

        info[:search_time_us] = CPUtime_us() - start_us
        info[:tree_queries] = i
        if p.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end
        
        best_Q = -Inf
        sanode = 0
        for child in banodes
            if tree.q[child] > best_Q
                best_Q = tree.q[child]
                sanode = child
            end
        end
        # XXX some publications say to choose action that has been visited the most
        a = tree.a_labels[sanode] # choose action with highest approximate value
    catch ex
        a = convert(POMDPs.action_type(p.mdp), default_action(p.solver.default_action, p.mdp, b, ex))
        info[:exception] = ex
    end

    return a, info
end

action(p::QBPlanner, b) = first(action_info(p, b))

function insert_action_orphan!{S,A}(tree::MCTS.DPWTree{S,A}, a::A, n0::Int, q0::Float64, maintain_a_lookup=true)
    push!(tree.n, n0)
    push!(tree.q, q0)
    push!(tree.a_labels, a)
    push!(tree.transitions, Vector{Tuple{Int,Float64}}[])
    sanode = length(tree.n)
    push!(tree.n_a_children, 0)
    @assert !maintain_a_lookup # not supported
    return sanode
end


