# Greedy vertex cover heuristic: iteratively selects the highest-degree vertex.

def greedy_vertex_cover(G):
    remaining_graph = G.copy()  ## Create a copy of the original graph
    cover = set()  ## Stores the vertices selected for the cover

    while remaining_graph.number_of_edges() > 0:
        v = max(remaining_graph.nodes, key=lambda item: remaining_graph.degree(item))  ## Select the vertex with highest degree
        cover.add(v)
        remaining_graph.remove_node(v)  ## Remove the selected vertex and all edges connected to it

    return cover