import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


myvector = pd.Series([10, 20, np.nan, 30, 40])

print(myvector.describe())
print(myvector.isnull().index)
