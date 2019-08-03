from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


# generate x feature encoder
encX = LabelEncoder()
encX.fit(['看電視', '讀書', '音樂', '游泳'])

# generate y feature encoder
ency = LabelEncoder()
ency.fit(['是', '否'])

# print(encX.classes_)

data_Xy = {'興趣':['看電視','讀書','音樂','看電視'],'成功與否':['是','否','否','是']}
df = pd.DataFrame(data= data_Xy, index=['小明','小林','小英','小陳'])
df = df[['興趣','成功與否']]
# print(df)

df_encode = df.copy()
df_encode['興趣'] = encX.transform(df_encode['興趣'])
df_encode['成功與否'] = ency.transform(df_encode['成功與否'])

print(df_encode)

prediction = np.array([1,0,0,1])
df['prediction'] = ency.inverse_transform(prediction) #將預測完的結果做反轉換

