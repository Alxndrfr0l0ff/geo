import geopandas as gpd
import matplotlib.pyplot as plt

# Завантажуємо GeoJSON
gdf = gpd.read_file('IF_reg_TG_boundary.geojson')

# Перевіряємо
print(gdf.head())

# Фільтруємо лише полігони
gdf_poly = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]

# Конвертуємо в Shapefile
gdf_poly.to_file('IF_reg_TG_bou_7.shp')

gdf_poly = gpd.read_file('IF_reg_TG_bou_7.shp')

# Візуалізація
ax = gdf_poly.plot(figsize=(10,10), edgecolor='black', linewidth=0.3)

# Додаємо назви (заміни 'name_uk' на свою колонку з назвами)
for idx, row in gdf_poly.iterrows():
    if row.geometry.is_empty:
        continue
    x, y = row.geometry.centroid.x, row.geometry.centroid.y
    plt.text(x, y, str(row['name_uk']), fontsize=8, ha='center')

plt.show()