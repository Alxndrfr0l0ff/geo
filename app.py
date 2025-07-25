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
gdf_poly.plot(figsize=(10,10), edgecolor='black', linewidth=0.3)
plt.show()