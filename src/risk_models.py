def sample_cov(returns_df, periods = 252):
    return returns_df.cov() * periods