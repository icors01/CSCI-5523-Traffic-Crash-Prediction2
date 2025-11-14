import geopandas as gpd
import numpy as np
import pandas as pd
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

    # Get gdf in metric crs
    gdf_m = gdf.to_crs(crs)

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

def reduce_gdf_area(gdf, min_lat, max_lat, min_long, max_long):
    """
    gdf: GeoDataFrame in EPSG:4326 (lat/lon)
    min_lat: minimum latitude bounds
    max_lat: maximum latitude bounds
    min_long: minimum longitude bounds
    max_long: maximum longitude bounds
    returns: GeoDataFrame only containing points within the bounds
    """
    return gdf[(gdf.geometry.y >= min_lat) & (gdf.geometry.y <= max_lat) & (gdf.geometry.x >= min_long) & (gdf.geometry.x <= max_long)]

def generate_grid_tc_metro_area(gdf, cell_size_m=500):
    """
    gdf: GeoDataFrame in EPSG:4326 (lat/lon)
    cell_size_m: grid cell size in meters
    returns: GeoDataFrame of grid cells in EPSG:4326, bounded around the Twin Cities metro area
    """

    # Remove all points outside the Twin Cities metro area (rough approximation of metro area as a rectangle and not including Wisconsin portions)
    gdf = reduce_gdf_area(gdf, 44.545707, 45.681479, -94.390340, -92.759918)

    # Call generate_grid with adjusted GDF
    return generate_grid(gdf, cell_size_m)

def create_cell_day_df(crash_gdf, grid_gdf, start_date, end_date):
    """
    crash_gdf: GeoDataFrame with crash information
    grid_gdf: GeoDataFrame with 'cell_id' and 'geometry' columns
    start_date: string or pd.Timestamp, date of earliest crash in return DataFrame (inclusive)
    end_date: strings or pd.Timestamp, dates of latest crashes in return DataFrame (exclusive - last day of crashes used for 'crash_tomorrow' label)
    returns: DataFrame with columns ['cell_id', 'date', 'num_crashes', 'crash_tomorrow']
    """

    # Convert DateOfIncident to datetime, truncate time (only worrying about date)
    crash_gdf['date'] = pd.to_datetime(crash_gdf['DateOfIncident']).dt.floor('D')

    # Filter crash_gdf to crashes occuring between start_date and end_date
    mask = (crash_gdf['date'] >= pd.to_datetime(start_date)) & (crash_gdf['date'] <= pd.to_datetime(end_date))
    crash_gdf = crash_gdf[mask].copy()

    # Make sure the CRS matches
    if crash_gdf.crs != grid_gdf.crs:
        crash_gdf = crash_gdf.to_crs(grid_gds.crs)

    # Assign each crash to a cell
    crash_with_cells = gpd.sjoin(crash_gdf, grid_gdf[['cell_id', 'geometry']], how='inner', predicate='within')

    # Aggregate crashes by cell/day combination
    crash_cells = (
        crash_with_cells
        .groupby(['cell_id', 'date'])
        .size()
        .reset_index(name='num_crashes')
    )

    # All possible cells and days
    all_cells = grid_gdf['cell_id'].unique()
    all_days = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create a DataFrame with each unique combination of cell and date
    full_index = pd.MultiIndex.from_product([all_cells, all_days], names=['cell_id', 'date'])
    full_df = pd.DataFrame(index=full_index).reset_index()

    # Add crash data to DataFrame with all cell and date combinations
    cell_day_df = full_df.merge(crash_cells, on=['cell_id', 'date'], how='left')
    cell_day_df['num_crashes'] = cell_day_df['num_crashes'].fillna(0).astype(int)

    # Sort by cell_id and date
    cell_day_df = cell_day_df.sort_values(['cell_id', 'date'])

    # Add crash_tomorrow classifier
    cell_day_df['crash_tomorrow'] = cell_day_df.groupby('cell_id')['num_crashes'].shift(-1)
    cell_day_df['crash_tomorrow'] = (cell_day_df['crash_tomorrow'] > 0).astype(int)

    # Remove data from the final day (as they all have crash_tommorow = 0 no matter what)
    cell_day_df = cell_day_df[cell_day_df['date'] < pd.to_datetime(end_date)]

    return cell_day_df
    

    
    


        