def is_vertex_cover(G, cover):
    """
    Checks whether 'cover' is a vertex cover of graph G.

    Parameters:
        G (networkx.Graph): input graph
        cover (iterable): collection of vertices

    Returns:
        bool: True if cover is a valid vertex cover, False otherwise
    """
    cover_set = set(cover) ## We covert the iterable into a set for fast membership testing

    for u, v in G.edges():
        if u not in cover_set and v not in cover_set: ## If neither ends of an edge is in the cover, then we don't have a vertex cover
            return False

    return True

def cover_size(cover):
    """
    Parameters:
        cover (iterable): collection of vertices

    Returns:
        int: the size of a vertex cover 
    """
    return len(set(cover))