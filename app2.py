import geopandas as gpd

gdf_boundaries = gpd.read_file("IF_reg_TG_bou_7.shp")
print(gdf_boundaries.columns)