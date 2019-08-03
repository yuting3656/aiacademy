from sklearn import preprocessing
import numpy as np

a = np.array([[10, 2.7, 3.6, 5],
              [-100, 5, -2, 10],
              [120, 20, 40, 50]], dtype=float)


def normalize(x, axis, method, minmax_range =(0,1)):
    if method == 'z-score':
        scale_a = preprocessing.scale(a, axis=axis)
        return scale_a
    elif method== 'minmax':
        scale_a = preprocessing.minmax_scale(a, axis=axis, feature_range=minmax_range) #default feature range 0~1
        return scale_a


# 改變axis，看看結果如何變化
axis =0
scale_a = normalize(a, axis, method = 'z-score')

# 改變axis，看看結果如何變化
axis =0
# 改變minmax_range看看結果如何變化
scale_a_2 = normalize(a, axis, method = 'minmax', minmax_range=(0,1))



