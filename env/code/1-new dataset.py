
import pandas as pd
import numpy as np
from netCDF4 import Dataset
import os

print("="*70)
print("READING WRF OUTPUT FILE - EXTRACTING ALL DATA")
print("="*70)

# ========== STEP 1: READ WRF FILE ==========
print("\nStep 1: Opening WRF output file...")

wrf_file = '../data/wrfout_d03_2017-08-25_00_00_00'
nc = Dataset(wrf_file, 'r')

print(f"✓ Opened: {wrf_file}")

# ========== STEP 2: LIST ALL VARIABLES ==========
print("\n" + "="*70)
print("AVAILABLE VARIABLES IN WRF FILE")
print("="*70)

print("\nAll variables in the file:")
for i, var_name in enumerate(nc.variables.keys(), 1):
    var = nc.variables[var_name]
    print(f"{i:3d}. {var_name:20s} - Shape: {var.shape} - {var.dimensions}")

# ========== STEP 3: EXTRACT KEY VARIABLES ==========
print("\n" + "="*70)
print("Step 3: Extracting variables...")
print("="*70)

# Get dimensions
times = nc.variables['Times'][:]
n_times = times.shape[0]
print(f"\nTime steps: {n_times}")

# Extract all 2D surface variables (Time, south_north, west_east)
variables_2d = {}

print("\nExtracting 2D surface variables:")
for var_name in nc.variables.keys():
    var = nc.variables[var_name]
    # Look for variables with 3 dimensions (Time, south_north, west_east)
    if len(var.shape) == 3 and var.shape[0] == n_times:
        try:
            variables_2d[var_name] = var[:]
            print(f"  ✓ {var_name:20s} - Shape: {var.shape}")
        except:
            print(f"  ✗ {var_name:20s} - Could not extract")

# Extract coordinate variables
print("\nExtracting coordinates:")
lon = nc.variables['XLONG'][:]
lat = nc.variables['XLAT'][:]
print(f"  ✓ LONGITUDE (XLONG) - Shape: {lon.shape}")
print(f"  ✓ LATITUDE (XLAT) - Shape: {lat.shape}")

# ========== STEP 4: CALCULATE DERIVED VARIABLES ==========
print("\nCalculating derived variables:")

# Wind speed and direction from U10 and V10
if 'U10' in variables_2d and 'V10' in variables_2d:
    u10 = variables_2d['U10']
    v10 = variables_2d['V10']
    
    wind_speed = np.sqrt(u10**2 + v10**2)
    wind_direction = (np.arctan2(u10, v10) * 180 / np.pi) % 360
    
    variables_2d['WIND_SPEED'] = wind_speed
    variables_2d['WIND_DIR'] = wind_direction
    print(f"  ✓ WIND_SPEED - Calculated from U10, V10")
    print(f"  ✓ WIND_DIR - Calculated from U10, V10")

# ========== STEP 5: FLATTEN DATA ==========
print("\n" + "="*70)
print("Step 5: Flattening data to table format...")
print("="*70)

# Get dimensions
if len(lon.shape) == 3:
    n_times, n_lat, n_lon = lon.shape
else:
    n_times = 1
    n_lat, n_lon = lon.shape

print(f"  Time steps: {n_times}")
print(f"  Grid points: {n_lat} × {n_lon} = {n_lat * n_lon:,}")
print(f"  Total data points: {n_times * n_lat * n_lon:,}")

# Create data list
data_list = []

print("\nFlattening arrays...")
for t in range(n_times):
    if t % 10 == 0:
        print(f"  Processing time step {t+1}/{n_times}...")
    
    for i in range(n_lat):
        for j in range(n_lon):
            data_point = {
                'time_step': t,
                'LONGITUDE': lon[t, i, j] if len(lon.shape) == 3 else lon[i, j],
                'LATITUDE': lat[t, i, j] if len(lat.shape) == 3 else lat[i, j],
            }
            
            # Add all 2D variables
            for var_name, var_data in variables_2d.items():
                try:
                    data_point[var_name] = var_data[t, i, j]
                except:
                    pass
            
            data_list.append(data_point)

df = pd.DataFrame(data_list)
print(f"\n✓ Created dataframe with {len(df):,} rows and {len(df.columns)} columns")

# Close netCDF file
nc.close()

# ========== STEP 6: DISPLAY DATA INFO ==========
print("\n" + "="*70)
print("DATA OVERVIEW")
print("="*70)

print(f"\nTotal columns: {len(df.columns)}")
print("\nColumns:", list(df.columns))

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nBasic Statistics (first few variables):")
print(df.describe())

# ========== STEP 7: CREATE GRID AND ASSIGN CELL IDs ==========
print("\n" + "="*70)
print("Step 7: Creating grid and assigning cell IDs...")
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

# ========== STEP 8: SAVE RESULTS ==========
print("\n" + "="*70)
print("Step 8: Saving results...")
print("="*70)



output_file = '../data/harvey_wrf_all_data_with_cells.csv'
df.to_csv(output_file, index=False)

print(f"\n✓ Saved to: {output_file}")
print(f"  Total rows: {len(df):,}")
print(f"  Total columns: {len(df.columns)}")
print(f"  File size: ~{len(df) * len(df.columns) * 8 / (1024**2):.1f} MB (estimated)")

# ========== STEP 9: SUMMARY STATISTICS ==========
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nPoints per cell:")
points_per_cell = df.groupby('cell_id').size()
print(f"  Mean: {points_per_cell.mean():.1f}")
print(f"  Median: {points_per_cell.median():.0f}")
print(f"  Min: {points_per_cell.min()}")
print(f"  Max: {points_per_cell.max()}")

print(f"\nCells with data: {df['cell_id'].nunique()} out of {n_cells*n_cells} possible cells")

if 'WIND_SPEED' in df.columns:
    print(f"\nWind Speed Statistics:")
    print(f"  Mean: {df['WIND_SPEED'].mean():.2f} m/s")
    print(f"  Max: {df['WIND_SPEED'].max():.2f} m/s")
    print(f"  Min: {df['WIND_SPEED'].min():.2f} m/s")

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print(f"\nAll WRF data saved to: {output_file}")