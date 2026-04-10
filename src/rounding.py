def threshold_rounding(x_values, threshold=0.5):
    """
    Converts a fractional LP solution into an integral vertex cover using threshold rounding.
    
    Parameters:
    x_values (dict): dictionary mapping each vertex to its LP value
    threshold (float): vertices with value >= threshold are included in the vertex cover
    Returns:
        list: vertices selected after rounding
    """
    rounded_cover = [] ## stores all the vertices selected after rounding 
    for v, value in x_values.items():
        if value >= threshold: ## Add the vertex to the cover if its LP value meets the threshold
            rounded_cover.append(v) 
             
    return rounded_cover