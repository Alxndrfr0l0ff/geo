import pandas as pd
import geopandas as gpd
from spreg import ML_Lag
from libpysal.weights import Queen
import numpy as np
import matplotlib.pyplot as plt

tourism = pd.read_csv('згруповано_турзбір.csv')
# Переконаємось у назвах колонок
print(tourism.columns)

ecol_atm = pd.read_excel('ecol_d1.xlsx')
ecol_wtr = pd.read_excel('ecol_d2.xlsx')
ecol_tpw = pd.read_excel('ecol_d3.xlsx')

print(ecol_atm.columns)

atm_agg = ecol_atm.groupby(['HKATOTTG', 'РІК'], as_index=False).agg({'ОБСЯГ_ВИКИДУ_Т': 'sum'})
atm_agg.rename(columns={'ОБСЯГ_ВИКИДУ_Т': 'Викиди_атмосфера_т'}, inplace=True)

wtr_agg = ecol_wtr.groupby(['HKATOTTG', 'РІК'], as_index=False).agg({'ОБСЯГ_ВИКИДУ_Т': 'sum'})
wtr_agg.rename(columns={'ОБСЯГ_ВИКИДУ_Т': 'Викиди_вода_т'}, inplace=True)

tpw_agg = ecol_tpw.groupby(['HKATOTTG', 'РІК'], as_index=False).agg({'ОБСЯГ_ВИКИДУ_Т': 'sum'})
tpw_agg.rename(columns={'ОБСЯГ_ВИКИДУ_Т': 'ТПВ_т'}, inplace=True)

# Злиття (по 'Код_громади', 'Рік')
df = tourism.merge(atm_agg, left_on=['Код_громади', 'Рік'], right_on=['HKATOTTG', 'РІК'], how='left') \
            .merge(wtr_agg, left_on=['Код_громади', 'Рік'], right_on=['HKATOTTG', 'РІК'], how='left', suffixes=('', '_wtr')) \
            .merge(tpw_agg, left_on=['Код_громади', 'Рік'], right_on=['HKATOTTG', 'РІК'], how='left', suffixes=('', '_tpw'))

print(df.head())

gdf = gpd.read_file('IF_reg_TG_bou_7.shp')
print(gdf.columns)

gdf_merged = gdf.merge(df, left_on='katotth', right_on='Код_громади', how='left')
gdf_2024 = gdf_merged[gdf_merged['Рік'] == 2024]
fig, ax = plt.subplots(figsize=(12, 10))
gdf_2024.plot(column='Викиди_атмосфера_т', cmap='OrRd', linewidth=0.3, edgecolor='black', legend=True, ax=ax)
ax.set_title("Туристо-доби за громадами (2024)", fontsize=16)
ax.axis('off')
plt.show()

corr = gdf_merged[['Всього_туристо_діб', 'Викиди_атмосфера_т', 'Викиди_вода_т', 'ТПВ_т']].corr()
print(corr)

gdf_2024 = gdf_merged[gdf_merged['Рік'] == 2024].copy()
gdf_2024 = gdf_2024.dropna(subset=['Всього_туристо_діб', 'Викиди_атмосфера_т', 'Викиди_вода_т', 'ТПВ_т'])

# Формуємо матрицю просторових ваг (сусідство Queen contiguity)
w = Queen.from_dataframe(gdf_2024)
w.transform = 'r'

# Вектор залежної змінної (туристо-доби)
y = gdf_2024['Всього_туристо_діб'].values.reshape(-1,1)

# Матриця незалежних змінних (екологічні показники)
X = gdf_2024[['Викиди_атмосфера_т', 'Викиди_вода_т', 'ТПВ_т']].values

# Просторова лагова регресія (Spatial Lag Model)
model = ML_Lag(y, X, w=w, name_y='Туристо-доби',
               name_x=['Атмосфера', 'Вода', 'ТПВ'],
               name_w='Сусідство')
print(model.summary)