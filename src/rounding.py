def threshold_rounding(x_values, threshold=0.5):
    """
    Converts a fractional LP solution into an integral vertex cover using threshold rounding.
    
    Parameters:
        x_values (dict): dictionary mapping each vertex to its LP value
        threshold (float): rounding threshold (default = 0.5 for this problem)
    Returns:
        list: vertices selected after rounding (vertex cover)
    """
    epsilon = 1e-9
    threshold = threshold - epsilon ## to avoid floating-point precision errors
    
    rounded_cover = {
    node for node, value in x_values.items() 
        if value >= (threshold)
    }
             
    return rounded_cover