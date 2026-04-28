import numpy as np

# Calculation of L_Aeq,T in accordance to ISO 1996-1:2017
def getL_Aeq(readings: np.ndarray, deltas):
    if (len(readings) != len(deltas)):
        raise ValueError("Input array lengths do not match.") 
    
    y = np.pow(10, readings / 10)
    t = np.cumsum(deltas)
    T = t[-1]

    result = np.trapezoid(y, t)
    result /= T
    result = 10 * np.log10(result)

    return result
