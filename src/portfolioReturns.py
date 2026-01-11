import numpy as np
import pandas as pd

def port_returns(returns_df, weights):
    w=np.array(list(weights.values()))
    return returns_df.dot(w)
