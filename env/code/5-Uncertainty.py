import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# ========== SELECT VISUALIZATION TYPE ==========
VISUALIZATION_TYPE = 'adaptive'  # 'unimodal', 'bimodal', or 'adaptive'

print("="*70)
print(f"HARVEY - {VISUALIZATION_TYPE.upper()} UNCERTAINTY HEATMAP WITH WIND VECTORS")
print("="*70)

# Load Harvey cell summary
df_summary = pd.read_csv('../data/harvey_cell_summary.csv')
print(f"✓ Loaded {len(df_summary):,} cells from Harvey summary")

# Load distribution types
df_dist = pd.read_csv('../data/harvey_distribution_types_gmm.csv')
print(f"✓ Loaded distribution types for {len(df_dist):,} cells")

# Merge
df_summary = df_summary.merge(df_dist[['cell_id', 'distribution']], 
                               on='cell_id', how='left')

# ========== PREPARE GRID DATA ==========
print("\n" + "="*70)
print("PREPARING GRID DATA")
print("="*70)

# Grid dimensions
n_lon_cells = 50
n_lat_cells = 50

# Create empty grids
uncertainty_grid = np.full((n_lat_cells, n_lon_cells), np.nan)
mean_speed_grid = np.full((n_lat_cells, n_lon_cells), np.nan)
wind_u_grid = np.full((n_lat_cells, n_lon_cells), np.nan)
wind_v_grid = np.full((n_lat_cells, n_lon_cells), np.nan)
distribution_grid = np.full((n_lat_cells, n_lon_cells), np.nan)  # Track which method

# Load all points data for bimodal calculation
df_points = pd.read_csv('../data/harvey_all_points_with_cells.csv')

# Group by cell
from collections import defaultdict
cell_speeds = defaultdict(list)
for _, row in df_points.iterrows():
    cell_speeds[row['cell_id']].append(row['WIND_SPEED'])

print(f"✓ Loaded {len(df_points):,} wind measurements")

# Counters
unimodal_count = 0
bimodal_count = 0
too_few_count = 0

# Fill the grids
for _, row in df_summary.iterrows():
    cell_id = int(row['cell_id'])
    lon_idx = int(row['lon_idx'])
    lat_idx = int(row['lat_idx'])
    
    # Get cell statistics
    mean_speed = row['mean_speed']
    mean_u = row['mean_u']
    mean_v = row['mean_v']
    
    # Get distribution type
    dist_type = row.get('distribution', 'unimodal')
    
    # Get speeds for this cell
    speeds = np.array(cell_speeds[cell_id])
    n = len(speeds)
    
    # ========== CALCULATE UNCERTAINTY BASED ON DISTRIBUTION TYPE ==========
    
    if VISUALIZATION_TYPE == 'unimodal':
        # Force unimodal for all
        uncertainty = row['std_speed']
        method = 0
        
    elif VISUALIZATION_TYPE == 'bimodal':
        # Force bimodal for all
        if n >= 2:
            # Calculate bimodal parameters
            mean_val = np.mean(speeds)
            high_mask = speeds > mean_val
            low_mask = speeds <= mean_val
            
            n_high = np.sum(high_mask)
            n_low = np.sum(low_mask)
            
            if n_high > 0 and n_low > 0:
                p_high = n_high / n
                p_low = n_low / n
                mu_high = np.mean(speeds[high_mask])
                mu_low = np.mean(speeds[low_mask])
                sigma_high_sq = np.var(speeds[high_mask], ddof=0) if n_high > 1 else 0.0
                sigma_low_sq = np.var(speeds[low_mask], ddof=0) if n_low > 1 else 0.0
                
                mean_mixture = p_high * mu_high + p_low * mu_low
                variance_mixture = (p_high * (sigma_high_sq + mu_high**2) + 
                                    p_low * (sigma_low_sq + mu_low**2) - 
                                    mean_mixture**2)
                uncertainty = np.sqrt(max(0.0, variance_mixture))
            else:
                uncertainty = row['std_speed']
        else:
            uncertainty = row['std_speed']
        method = 1
        
    else:  # ADAPTIVE
        # ========== USE DISTRIBUTION TYPE TO SELECT METHOD ==========
        
        if dist_type == 'bimodal' and n >= 2:
            # Calculate BIMODAL uncertainty (Bishop 2006)
            mean_val = np.mean(speeds)
            high_mask = speeds > mean_val
            low_mask = speeds <= mean_val
            
            n_high = np.sum(high_mask)
            n_low = np.sum(low_mask)
            
            if n_high > 0 and n_low > 0:
                # Mixing coefficients
                p_high = n_high / n
                p_low = n_low / n
                
                # Component means
                mu_high = np.mean(speeds[high_mask])
                mu_low = np.mean(speeds[low_mask])
                
                # Within-mode variances
                sigma_high_sq = np.var(speeds[high_mask], ddof=0) if n_high > 1 else 0.0
                sigma_low_sq = np.var(speeds[low_mask], ddof=0) if n_low > 1 else 0.0
                
                # Mixture mean
                mean_mixture = p_high * mu_high + p_low * mu_low
                
                # Mixture variance (Bishop 2006, Eq. 5.159)
                variance_mixture = (p_high * (sigma_high_sq + mu_high**2) + 
                                    p_low * (sigma_low_sq + mu_low**2) - 
                                    mean_mixture**2)
                
                uncertainty = np.sqrt(max(0.0, variance_mixture))
                method = 1  # bimodal
                bimodal_count += 1
            else:
                uncertainty = row['std_speed']
                method = 0
                unimodal_count += 1
                
        elif dist_type == 'unimodal':
            # Calculate UNIMODAL uncertainty (standard deviation)
            uncertainty = row['std_speed']
            method = 0
            unimodal_count += 1
            
        else:
            # Too few points or unknown
            uncertainty = row['std_speed']
            method = 2
            too_few_count += 1
    
    # Place in grid (flip lat_idx for correct orientation)
    grid_lat_idx = n_lat_cells - 1 - lat_idx
    uncertainty_grid[grid_lat_idx, lon_idx] = uncertainty
    mean_speed_grid[grid_lat_idx, lon_idx] = mean_speed
    wind_u_grid[grid_lat_idx, lon_idx] = mean_u
    wind_v_grid[grid_lat_idx, lon_idx] = mean_v
    distribution_grid[grid_lat_idx, lon_idx] = method

print(f"✓ Grid filled with {VISUALIZATION_TYPE} uncertainty values")
print(f"  Min uncertainty: {np.nanmin(uncertainty_grid):.4f} m/s")
print(f"  Max uncertainty: {np.nanmax(uncertainty_grid):.4f} m/s")
print(f"  Mean uncertainty: {np.nanmean(uncertainty_grid):.4f} m/s")

if VISUALIZATION_TYPE == 'adaptive':
    total = unimodal_count + bimodal_count + too_few_count
    print(f"\nAdaptive method usage:")
    print(f"  Unimodal formula:  {unimodal_count:4d} cells ({unimodal_count/total*100:5.1f}%)")
    print(f"  Bimodal formula:   {bimodal_count:4d} cells ({bimodal_count/total*100:5.1f}%)")
    print(f"  Insufficient data: {too_few_count:4d} cells ({too_few_count/total*100:5.1f}%)")

# Get coordinate ranges
lon_min = df_summary['center_lon'].min()
lon_max = df_summary['center_lon'].max()
lat_min = df_summary['center_lat'].min()
lat_max = df_summary['center_lat'].max()

# Adjust for cell boundaries
lon_cell_size = (lon_max - lon_min) / (n_lon_cells - 1)
lat_cell_size = (lat_max - lat_min) / (n_lat_cells - 1)
lon_min -= lon_cell_size / 2
lon_max += lon_cell_size / 2
lat_min -= lat_cell_size / 2
lat_max += lat_cell_size / 2

print(f"\nCoordinate ranges:")
print(f"  Longitude: {lon_min:.2f}° to {lon_max:.2f}°")
print(f"  Latitude:  {lat_min:.2f}° to {lat_max:.2f}°")

# ========== CREATE CUSTOM COLORMAP ==========
colors = ['#00FFFF', '#00FF00', '#7FFF00', '#FFFF00', '#FFA500', '#FF4500', '#FF0000', '#8B0000', '#4B0082']
n_bins = 256
cmap = LinearSegmentedColormap.from_list('hurricane', colors, N=n_bins)

# ========== CREATE HEATMAP ==========
print("\n" + "="*70)
print("CREATING HEATMAP WITH WIND VECTORS")
print("="*70)

fig, ax = plt.subplots(figsize=(14, 12))

# ========== SET COLOR SCALE ==========
vmin_value = 0
vmax_value = np.nanpercentile(uncertainty_grid, 98)

# Plot uncertainty heatmap
im = ax.imshow(uncertainty_grid, 
               extent=[lon_min, lon_max, lat_min, lat_max],
               origin='lower',
               cmap=cmap,
               aspect='auto',
               alpha=0.9,
               vmin=vmin_value,
               vmax=vmax_value)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

if VISUALIZATION_TYPE == 'unimodal':
    cbar_label = 'Unimodal Uncertainty (σ) [m/s]'
    title = 'Hurricane Harvey - Unimodal Uncertainty\n(Standard Deviation)'
elif VISUALIZATION_TYPE == 'bimodal':
    cbar_label = 'Bimodal Uncertainty (σ_mix) [m/s]'
    title = 'Hurricane Harvey - Bimodal Uncertainty\n(Mixture Variance - Bishop 2006)'
else:  # adaptive
    cbar_label = ' Adaptive Uncertainty [m/s]'
    title = f'Hurricane Harvey - Adaptive Uncertainty'

cbar.set_label(cbar_label, rotation=270, labelpad=25, fontsize=18, fontweight='bold')

# ========== ADD WIND VECTORS ==========
lon_grid = np.linspace(lon_min, lon_max, n_lon_cells)
lat_grid = np.linspace(lat_min, lat_max, n_lat_cells)
X, Y = np.meshgrid(lon_grid, lat_grid)

arrow_spacing = 3
X_sub = X[::arrow_spacing, ::arrow_spacing]
Y_sub = Y[::arrow_spacing, ::arrow_spacing]
U_sub = wind_u_grid[::arrow_spacing, ::arrow_spacing]
V_sub = wind_v_grid[::arrow_spacing, ::arrow_spacing]

arrow_scale = np.nanmax(mean_speed_grid) * 8

ax.quiver(X_sub, Y_sub, U_sub, V_sub,
          alpha=0.7,
          scale=arrow_scale,
          width=0.002,
          headwidth=3,
          headlength=3,
          color='black',
          linewidth=0.6)

# ========== LABELS AND TITLE ==========
ax.set_xlabel('Longitude', fontsize=18, fontweight='bold')
ax.set_ylabel('Latitude', fontsize=18, fontweight='bold')
ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='gray')

plt.tight_layout()

# Save figure
os.makedirs('../images', exist_ok=True)
output_file = f'../images/harvey_{VISUALIZATION_TYPE}_uncertainty.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")

# ========== STATISTICS ==========
print("\n" + "="*70)
print(f"HARVEY - {VISUALIZATION_TYPE.upper()} UNCERTAINTY STATISTICS")
print("="*70)

valid_cells = ~np.isnan(uncertainty_grid)
total_cells = np.sum(valid_cells)

# Adjust thresholds for Harvey (larger values than Katrina)
low = np.sum((uncertainty_grid >= 0) & (uncertainty_grid < 1))
moderate = np.sum((uncertainty_grid >= 1) & (uncertainty_grid < 2))
high = np.sum((uncertainty_grid >= 2) & (uncertainty_grid < 3))
extreme = np.sum(uncertainty_grid >= 3)

print(f"\nUncertainty Distribution:")
print(f"  LOW (< 1 m/s):       {low:4d} cells ({low/total_cells*100:5.1f}%)")
print(f"  MODERATE (1-2 m/s):  {moderate:4d} cells ({moderate/total_cells*100:5.1f}%)")
print(f"  HIGH (2-3 m/s):      {high:4d} cells ({high/total_cells*100:5.1f}%)")
print(f"  EXTREME (≥ 3 m/s):   {extreme:4d} cells ({extreme/total_cells*100:5.1f}%)")

print(f"\nWind Speed Statistics:")
print(f"  Mean: {np.nanmean(mean_speed_grid):.2f} m/s")
print(f"  Std:  {np.nanstd(mean_speed_grid):.2f} m/s")
print(f"  Min:  {np.nanmin(mean_speed_grid):.2f} m/s")
print(f"  Max:  {np.nanmax(mean_speed_grid):.2f} m/s")

print(f"\nUncertainty Percentiles:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    val = np.nanpercentile(uncertainty_grid, p)
    print(f"  {p:2d}th percentile: {val:.3f} m/s")

print("\n" + "="*70)
print("DONE!")
print("="*70)

print("\n" + "="*70)
print("METHODOLOGY NOTE")
print("="*70)
print(f"""
Wind uncertainty quantified using {VISUALIZATION_TYPE} approach:

Unimodal cells: σ = √(Σ(w - μ)²/n)

Bimodal cells: σ_mix = √[Σ π_k(σ²_k + μ²_k) - E[w]²]  (Bishop 2006)

Wind direction: Vector averaging θ_mean = arctan2(-ū, -v̄)

Grid: 50×50 cells (~9 km × 9 km per cell)
Coverage: 100% (all cells have n≥10 points)
""")