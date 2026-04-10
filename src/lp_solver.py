import pulp as p

def solve_vertex_cover_lp(G):
    """
    Solve the LP relaxation of the Minimum Vertex Cover problem.

    Parameters:
        G (networkx.Graph): input graph

    Returns:
        dict: {
            "status": str,
            "objective": float,
            "x_values": dict
        }
    """
    Lp_prob = p.LpProblem('Vertex_Cover_LP', p.LpMinimize) 

    x = {
        v: p.LpVariable(f"x_{v}", lowBound=0, upBound=1, cat="Continuous")
        for v in G.nodes()
    }

    Lp_prob += p.lpSum(x[v] for v in G.nodes()), "Minimize_Vertex_Cover_Size"

    for u, v in G.edges():
        Lp_prob += x[u] + x[v] >= 1, f"cover_edge_{u}_{v}"

    Lp_prob.solve(p.PULP_CBC_CMD(msg = False))
    
    status = p.LpStatus[Lp_prob.status]
    objective = p.value(Lp_prob.objective)
    x_values = {v: p.value(x[v]) for v in G.nodes()}

    return {
        "status": status,
        "objective": objective,
        "x_values": x_values,
    }