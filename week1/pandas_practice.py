import pandas as pd
import numpy as np

df = pd.read_csv('03_pandaserercies.csv', index_col=0)

# 轉成str
df['date'] = df['date'].astype(str)
# 未滿4位數，位數補零
df['Time'] = df['Time'].apply(lambda x: '{:0>4}'.format(x))
# 組合時間字串
date_string = df['date'] + df['Time']
# 換成pandas time type
date_times = pd.to_datetime(date_string, format='%Y%m%d%H%M')
# 塞到index
df_clean = df.set_index(date_times)

# print(df_clean.head())
# print(df_clean['Time'])
# print(df_clean.info)

# 看選取範圍
print(df_clean.loc['2011-06-20 08:00:00':'2011-06-20 10:00:00', 'dry_bulb_faren'])
"""
If 'raise', then invalid parsing will raise an exception
If 'coerce', then invalid parsing will be set as NaN
If 'ignore', then invalid parsing will return the input 
"""
df_clean['dry_bulb_faren'] = pd.to_numeric(df_clean['dry_bulb_faren'], errors = 'coerce')
print(df_clean.loc['2011-06-20 08:00:00':'2011-06-20 10:00:00', 'dry_bulb_faren'])

#
df_clean['wind_speed'] = pd.to_numeric(df_clean["wind_speed"], errors="coerce")
df_clean['dew_point_faren'] = pd.to_numeric(df_clean["dew_point_faren"], errors="coerce")


# resample
daily_mean_2011 = df_clean.resample("D").mean()
daily_temp_2011 = daily_mean_2011["dry_bulb_faren"].values
print(daily_mean_2011.head())
