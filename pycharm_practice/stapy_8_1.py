import pandas as pd
from sklearn.preprocessing import scale
import numpy as np

cellraw = pd.read_csv('Data/Data_3/trad_alpha103.txt', header=0, index_col= 0, sep='\t')
print(cellraw)
