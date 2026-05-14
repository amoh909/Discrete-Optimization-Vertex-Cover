# Utility functions for validating and measuring vertex covers.

def is_vertex_cover(G, cover):

    cover_set = set(cover) ## We covert the iterable into a set for fast membership testing

    for u, v in G.edges():
        if u not in cover_set and v not in cover_set: ## If neither ends of an edge is in the cover, then we don't have a vertex cover
            return False

    return True

def cover_size(cover):
    return len(set(cover))