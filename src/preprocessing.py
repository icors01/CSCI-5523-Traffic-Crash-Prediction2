import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

def generate_grid(gdf, cell_size_m=5000, crs="EPSG:5070"):
    """
    gdf: GeoDataFrame in EPSG:4326 (lat/lon)
    cell_size_m: grid cell size in meters
    crs: metric CRS to build grid in
    returns: GeoDataFrame of grid cells in EPSG:4326 (lat/lon)
    """

    # Set crs if GeoDataFrame doesn't already contain it
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # Get gdf in metri crs
    gdf_m = gdf.to_crs(crs);

    # Set minimum and maximum bounds
    minx, miny, maxx, maxy = gdf_m.total_bounds
    minx -= cell_size_m
    miny -= cell_size_m
    maxx += cell_size_m
    maxy += cell_size_m

    # Get gridlines
    xs = np.arange(minx, maxx, cell_size_m)
    ys = np.arange(miny, maxy, cell_size_m)

    # Generate cells
    cells = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
                square = Polygon([
                    (x, y),
                    (x + cell_size_m, y),
                    (x + cell_size_m, y + cell_size_m),
                    (x, y + cell_size_m)
                ])
                cells.append({"cell_id": f"{i}_{j}", "geometry": square})
            
    # Generate the GeoDataFrame
    grid_m = gpd.GeoDataFrame(cells, crs=crs)

    # Converte back to EPSG:4326 (lat/long)
    grid = grid_m.to_crs("EPSG:4326")
    
    return grid
        