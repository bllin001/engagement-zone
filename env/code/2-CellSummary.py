import pandas as pd
import numpy as np
from netCDF4 import Dataset
import os

print("="*70)
print("HURRICANE HARVEY WRF DATA - ESSENTIAL VARIABLES + GRIDDING")
print("="*70)

# ========== STEP 1: READ WRF FILE ==========
print("\nStep 1: Opening WRF output file...")

wrf_file = '../data/wrfout_d03_2017-08-25_00_00_00'
nc = Dataset(wrf_file, 'r')

print(f"✓ Opened: {wrf_file}")

# ========== STEP 2: EXTRACT ESSENTIAL VARIABLES ==========
print("\nStep 2: Extracting essential variables...")


# Coordinates
lon = nc.variables['XLONG'][:]
lat = nc.variables['XLAT'][:]

# Wind components
u10 = nc.variables['U10'][:]
v10 = nc.variables['V10'][:]

# Atmospheric conditions
t2 = nc.variables['T2'][:]  # Temperature at 2m
psfc = nc.variables['PSFC'][:]  # Surface pressure
q2 = nc.variables['Q2'][:]  # Water vapor mixing ratio

# Terrain
hgt = nc.variables['HGT'][:]  # Terrain height
landmask = nc.variables['LANDMASK'][:]  # Land/water mask

print("✓ Extracted variables:")
print(f"  - XLONG, XLAT (coordinates)")
print(f"  - U10, V10 (wind components)")
print(f"  - T2, PSFC, Q2 (atmospheric)")
print(f"  - HGT, LANDMASK (terrain)")

# ========== STEP 3: CALCULATE WIND SPEED AND DIRECTION ==========
print("\nStep 3: Calculating wind speed and direction...")

wind_speed = np.sqrt(u10**2 + v10**2)
wind_direction = (np.arctan2(u10, v10) * 180 / np.pi) % 360

print("✓ Calculated WIND_SPEED and WIND_DIR")

# Close netCDF file
nc.close()

# ========== STEP 4: FLATTEN DATA ==========
print("\nStep 4: Flattening data to table format...")

# Get dimensions
n_times, n_lat, n_lon = wind_speed.shape

print(f"  Time steps: {n_times}")
print(f"  Grid points: {n_lat} × {n_lon} = {n_lat * n_lon:,}")
print(f"  Total data points: {n_times * n_lat * n_lon:,}")

# Create data list
data_list = []

print("\nProcessing time steps...")
for t in range(n_times):
    print(f"  Time step {t+1}/{n_times}...", end='\r')
    
    for i in range(n_lat):
        for j in range(n_lon):
            data_list.append({
                'time_step': t,
                'LONGITUDE': lon[t, i, j],
                'LATITUDE': lat[t, i, j],
                'U10': u10[t, i, j],
                'V10': v10[t, i, j],
                'WIND_SPEED': wind_speed[t, i, j],
                'WIND_DIR': wind_direction[t, i, j],
                'T2': t2[t, i, j],
                'PSFC': psfc[t, i, j],
                'Q2': q2[t, i, j],
                'HGT': hgt[t, i, j],
                'LANDMASK': landmask[t, i, j]
            })

df = pd.DataFrame(data_list)
print(f"\n✓ Created dataframe with {len(df):,} rows and {len(df.columns)} columns")

# ========== STEP 5: CREATE GRID AND ASSIGN CELL IDs ==========
print("\n" + "="*70)
print("Step 5: Creating grid and assigning cell IDs...")
print("="*70)

n_cells = 50  # 50×50 grid

# Get data range
lon_min = df['LONGITUDE'].min()
lon_max = df['LONGITUDE'].max()
lat_min = df['LATITUDE'].min()
lat_max = df['LATITUDE'].max()

print(f"\nDomain:")
print(f"  Longitude: [{lon_min:.2f}° to {lon_max:.2f}°]")
print(f"  Latitude:  [{lat_min:.2f}° to {lat_max:.2f}°]")

# Create bins
lon_bins = np.linspace(lon_min, lon_max, n_cells + 1)
lat_bins = np.linspace(lat_min, lat_max, n_cells + 1)

# Calculate cell size
lon_cell_size = (lon_max - lon_min) / n_cells
lat_cell_size = (lat_max - lat_min) / n_cells
cell_size_km = lat_cell_size * 111  # Approximate conversion to km

print(f"\nGrid Configuration:")
print(f"  Grid size: {n_cells}×{n_cells} = {n_cells*n_cells:,} total cells")
print(f"  Cell size: {lon_cell_size:.3f}° × {lat_cell_size:.3f}°")
print(f"  Cell size: ~{cell_size_km:.1f} km × ~{cell_size_km:.1f} km")

# Assign points to cells
lon_indices = np.clip(np.digitize(df['LONGITUDE'], lon_bins) - 1, 0, n_cells - 1)
lat_indices = np.clip(np.digitize(df['LATITUDE'], lat_bins) - 1, 0, n_cells - 1)

# Add grid information to dataframe
df['lon_idx'] = lon_indices
df['lat_idx'] = lat_indices
df['cell_id'] = lat_indices * n_cells + lon_indices

# Calculate cell center coordinates
df['cell_center_lon'] = (lon_bins[df['lon_idx']] + lon_bins[df['lon_idx'] + 1]) / 2
df['cell_center_lat'] = (lat_bins[df['lat_idx']] + lat_bins[df['lat_idx'] + 1]) / 2

print(f"\n✓ Assigned {len(df):,} points to {df['cell_id'].nunique()} unique cells")

# ========== STEP 6: SAVE ALL POINTS WITH CELL IDs ==========
print("\n" + "="*70)
print("Step 6: Saving all points with cell IDs...")
print("="*70)


output_all_points = '../data/harvey_all_points_with_cells.csv'
df.to_csv(output_all_points, index=False)

print(f"\n✓ Saved: {output_all_points}")
print(f"  Total rows: {len(df):,}")
print(f"  Total columns: {len(df.columns)}")
print(f"  Columns: {list(df.columns)}")

# ========== STEP 7: CALCULATE CELL STATISTICS ==========
print("\n" + "="*70)
print("Step 7: Calculating statistics for each cell...")
print("="*70)

# Group by cell_id and calculate statistics
cell_stats_list = []

for cell_id in df['cell_id'].unique():
    cell_data = df[df['cell_id'] == cell_id]
    
    # Get cell indices and center
    lon_idx = cell_data['lon_idx'].iloc[0]
    lat_idx = cell_data['lat_idx'].iloc[0]
    center_lon = cell_data['cell_center_lon'].iloc[0]
    center_lat = cell_data['cell_center_lat'].iloc[0]
    
    # Number of points in cell
    n_points = len(cell_data)
    
    # Wind speed statistics
    wind_speeds = cell_data['WIND_SPEED'].values
    mean_speed = np.mean(wind_speeds)
    std_speed = np.std(wind_speeds, ddof=0) if n_points > 1 else 0
    median_speed = np.median(wind_speeds)
    min_speed = np.min(wind_speeds)
    max_speed = np.max(wind_speeds)
    
    # Wind direction - vector averaging
    directions_rad = np.deg2rad(cell_data['WIND_DIR'].values)
    u_comp = -wind_speeds * np.sin(directions_rad)
    v_comp = -wind_speeds * np.cos(directions_rad)
    mean_u = np.mean(u_comp)
    mean_v = np.mean(v_comp)
    mean_direction = np.rad2deg(np.arctan2(-mean_u, -mean_v)) % 360
    
    # Atmospheric conditions
    mean_temp = np.mean(cell_data['T2'].values)
    mean_pressure = np.mean(cell_data['PSFC'].values)
    mean_humidity = np.mean(cell_data['Q2'].values)
    
    # Terrain
    mean_height = np.mean(cell_data['HGT'].values)
    landmask_value = cell_data['LANDMASK'].iloc[0]  # Should be constant
    
    cell_stats_list.append({
        'cell_id': cell_id,
        'lon_idx': lon_idx,
        'lat_idx': lat_idx,
        'center_lon': center_lon,
        'center_lat': center_lat,
        'n_points': n_points,
        
        # Wind speed statistics
        'mean_speed': mean_speed,
        'std_speed': std_speed,
        'median_speed': median_speed,
        'min_speed': min_speed,
        'max_speed': max_speed,
        
        # Wind direction
        'mean_direction': mean_direction,
        'mean_u': mean_u,
        'mean_v': mean_v,
        
        # Atmospheric
        'mean_temperature': mean_temp,
        'mean_pressure': mean_pressure,
        'mean_humidity': mean_humidity,
        
        # Terrain
        'mean_height': mean_height,
        'landmask': landmask_value
    })

df_cells = pd.DataFrame(cell_stats_list)
print(f"✓ Calculated statistics for {len(df_cells)} cells")

# ========== STEP 8: SAVE CELL SUMMARY ==========
print("\nStep 8: Saving cell summary...")

output_cells = '../data/harvey_cell_summary.csv'
df_cells.to_csv(output_cells, index=False)

print(f"\n✓ Saved: {output_cells}")
print(f"  Total cells: {len(df_cells)}")
print(f"  Columns: {list(df_cells.columns)}")

# ========== STEP 9: FINAL SUMMARY ==========
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print(f"\nFile 1: {output_all_points}")
print(f"  - Contains ALL {len(df):,} data points")
print(f"  - Each point has its cell_id assignment")
print(f"  - Columns: {len(df.columns)}")

print(f"\nFile 2: {output_cells}")
print(f"  - Contains {len(df_cells)} cell summaries")
print(f"  - Statistics aggregated per cell")
print(f"  - Columns: {len(df_cells.columns)}")

print(f"\nPoints per cell:")
print(f"  Mean: {df_cells['n_points'].mean():.1f}")
print(f"  Median: {df_cells['n_points'].median():.0f}")
print(f"  Min: {df_cells['n_points'].min()}")
print(f"  Max: {df_cells['n_points'].max()}")

print(f"\nWind Speed Statistics (across all cells):")
print(f"  Mean: {df_cells['mean_speed'].mean():.2f} m/s")
print(f"  Max: {df_cells['mean_speed'].max():.2f} m/s")
print(f"  Min: {df_cells['mean_speed'].min():.2f} m/s")

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print(f"\n✓ All points saved to: {output_all_points}")
print(f"✓ Cell summary saved to: {output_cells}")