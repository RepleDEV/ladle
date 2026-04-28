import numpy as np
import pandas as pd

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

def processValues(df: pd.DataFrame):
    timestamps = pd.to_datetime(df["timestamp"]).to_list()
    
    # Calculate timestamp_deltas list
    # Where deltas[i] = timestamps[i + 1] - timestamps[i] for i = 0, 1, 2, ... len(timestamps)
    timestamp_deltas = [0]
    for i in range(1, len(timestamps)):
        now = timestamps[i]
        prev = timestamps[i - 1]
        delta_ms = (now - prev).microseconds // 1000

        timestamp_deltas.append(delta_ms)

    # timestamp_mean = np.mean(np.array(timestamp_deltas))

    readings = df["reading"].to_numpy()

    # Calculate index where the cumulative sum of deltas equals to T_ms
    delta_cumsum = np.cumsum(np.array(timestamp_deltas))
    T_ms = 5 * 60 * 1000 # TODO: Don't hardcode this
    total_indexes = np.argmax(delta_cumsum > T_ms) + 1

    readings = readings[:total_indexes]
    deltas = timestamp_deltas[:total_indexes]

    result = {}

    result["L_Aeq"] = getL_Aeq(readings, deltas);
    result["L_10"] = np.percentile(readings, 90)
    result["L_90"] = np.percentile(readings, 10)
    result["L_min"] = np.min(readings)
    result["L_max"] = np.max(readings)

    return result

# Calculating L_day^c from Brink et al. 2018
def getL_den(L_Aeq_day: np.ndarray, L_Aeq_eve: np.ndarray):
    # 7 for 7 hours from 0700 to 1400
    L_Aeq_day = np.pow(10, L_Aeq_day / 10) * 7
    # 5 for 5 hours from 1400 to 1900
    L_Aeq_eve = np.pow(10, L_Aeq_eve / 10) * 5

    L_day = 10 * np.log10((L_Aeq_day + L_Aeq_eve) / 12)
    L_den = L_day + 1.5

    return L_den 

def getL_TNI(
        L_90_day: np.ndarray, 
        L_90_eve: np.ndarray, 
        L_10_day: np.ndarray, 
        L_10_eve: np.ndarray
        ):
    L_90_day = np.pow(10, L_90_day / 10) * 7
    L_10_day = np.pow(10, L_10_day / 10) * 7
    L_90_eve = np.pow(10, L_90_eve / 10) * 7
    L_10_eve = np.pow(10, L_10_eve / 10) * 7

    L_90 = 10 * np.log10((L_90_day + L_90_eve) / 12)
    L_10 = 10 * np.log10((L_10_day + L_10_eve) / 12)

    L_TNI = 4 * (L_10 - L_90) + L_90 - 30

    return L_TNI
