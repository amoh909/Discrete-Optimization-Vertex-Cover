import pulp as p

def solve_vertex_cover_ilp(G): #Generally used on small graphs to have a baseline to compare with
    """
    Solve the ILP Minimum Vertex Cover problem.

    Parameters:
        G (networkx.Graph): input graph

    Returns:
        dict: {
            "status": str,
            "objective": float,
            "x_values": dict
            "cover" set 
        }
    """
    ILp_prob = p.LpProblem('Vertex_Cover_ILP', p.LpMinimize) 

    x = {
        v: p.LpVariable(f"x_{v}", cat="Binary")
        for v in G.nodes()
    }

    ILp_prob += p.lpSum(x[v] for v in G.nodes()), "Minimize_Vertex_Cover_Size"

    for u, v in G.edges():
        ILp_prob += x[u] + x[v] >= 1, f"cover_edge_{u}_{v}"

    ILp_prob.solve(p.PULP_CBC_CMD(msg = False))
    
    status = p.LpStatus[ILp_prob.status]
    objective = p.value(ILp_prob.objective)
    x_values = {v: p.value(x[v]) for v in G.nodes()}
    cover = {v for v in G.nodes() if x_values[v] == 1} ## Makes sense here to have a cover unlike the case where we had a relaxed Lp problem

    return {
        "status": status,
        "objective": objective,
        "x_values": x_values,
        "cover": cover
    }