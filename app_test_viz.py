import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Завантаження геопросторових даних громад (shapefile)
gdf_boundaries = gpd.read_file("IF_reg_TG_bou_7.shp")

df_grouped = pd.read_csv("згруповано_турзбір.csv")
# Фільтруємо дані за потрібним роком
df_2024 = df_grouped[df_grouped['Рік'] == 2024]

# Об'єднуємо геодані з табличними за кодом громади
gdf_merged = gdf_boundaries.merge(df_2024, left_on='katotth', right_on='Код_громади')

# Візуалізація кількості туристо-діб
fig, ax = plt.subplots(figsize=(10, 10))
gdf_merged.plot(column='Всього_туристо_діб',
                ax=ax,
                cmap='OrRd',
                legend=True,
                legend_kwds={'label': "Кількість туристо-діб"},
                edgecolor='black',
                linewidth=0.2)

ax.set_title('Кількість туристо-діб за громадами (2024)', fontsize=14)
ax.axis('off')
plt.show()
