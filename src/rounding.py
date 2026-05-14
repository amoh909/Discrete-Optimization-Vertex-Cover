# Threshold rounding: converts fractional LP solution to integral vertex cover at x >= 0.5.

def threshold_rounding(x_values, threshold=0.5):

    epsilon = 1e-9
    threshold = threshold - epsilon ## to avoid floating-point precision errors
    
    rounded_cover = {
    node for node, value in x_values.items() 
        if value >= (threshold)
    }
             
    return rounded_cover