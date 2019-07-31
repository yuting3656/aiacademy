import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

binomial_rvs = stats.binom.rvs(10, 0.8, size = 100)
pd.Series(binomial_rvs).value_counts()
print(pd.Series(binomial_rvs).value_counts())