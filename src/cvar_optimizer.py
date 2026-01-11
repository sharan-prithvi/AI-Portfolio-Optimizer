import cvxpy as cp
import numpy as np
import pandas as pd

class CVaROpt:
    def __init__(self, returns: pd.DataFrame, confidence: float = 0.95):
        self.returns = returns
        self.confidence = confidence
        self.n_assets = returns.shape[1]
        self.n_obs = returns.shape[0]
        
    def optimize(self):
        w = cp.Variable(self.n_assets)
        alpha = cp.Variable()
        z= cp.Variable(self.n_obs)
        returns_mat = self.returns.values
        port_returns = returns_mat @ w
        losses = -port_returns
        objective = cp.Minimize(
            alpha + (1 / ((1 - self.confidence) * self.n_obs)) * cp.sum(z)
        )
        constraints = [
            z >= losses - alpha,
            z >= 0,
            cp.sum(w) == 1,
            w >= 0
        ]
        problem = cp.Problem(objective=objective, constraints=constraints)
        problem.solve()
        return pd.Series(w.value, index = self.returns.columns)
    