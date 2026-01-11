import numpy as np
import pandas as pd

def sample_cov(returns_df, periods = 252):
    return returns_df.cov() * periods

def ewma_cov(returns_df, lambda_=0.94):
    returns = returns_df.values
    T, N = returns.shape
    
    cov = np.zeros((N,N))
    
    for t in range(T):
        r_t = returns[t].reshape(-1,1)
        cov = lambda_ * cov + (1-lambda_)*(r_t @ r_t.T)
    
    cov_annualized= cov * 252
    
    return pd.DataFrame(
        cov_annualized,
        index=returns_df.columns,
        columns=returns_df.columns
    )

def shrinkage_cov(sample_cov, delta=0.2):
    if not 0 <= delta <=1:
        raise ValueError("delta must be between 0 and 1")
    
    S= sample_cov.values
    
    F= np.diag(np.diag(S))
    
    shrunk_cov= delta * F + (1-delta) * S
    
    return pd.DataFrame(
        shrunk_cov,
        index=sample_cov.index,
        columns=sample_cov.columns
    )