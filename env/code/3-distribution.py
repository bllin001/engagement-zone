import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from collections import defaultdict

print("="*70)
print("HARVEY WRF - GMM DISTRIBUTION DETECTION FOR WIND SPEED")
print("="*70)

# ========== LOAD DATA ==========
print("\nLoading Harvey WRF data with cell assignments...")

df = pd.read_csv('../data/harvey_all_points_with_cells.csv')
print(f"✓ Loaded {len(df):,} data points")
print(f"✓ Grid already assigned: {df['cell_id'].nunique()} unique cells")

# ========== GROUP BY CELL ==========
print("\nGrouping by cell...")

cell_data = defaultdict(list)
for _, row in df.iterrows():
    cell_data[row['cell_id']].append(row['WIND_SPEED'])

print(f"✓ Grouped into {len(cell_data)} cells")

# ========== DETECT DISTRIBUTION TYPE USING GMM ==========
print("\nDetecting distributions using Gaussian Mixture Models...")

results = []

for cell_id, speeds in cell_data.items():
    speeds_array = np.array(speeds).reshape(-1, 1)
    n = len(speeds_array)
    
    if n < 10:
        dist_type = 'too_few_points'
        bic1 = np.nan
        bic2 = np.nan
        delta_bic = np.nan
    else:
        # Fit GMM with 1 component (unimodal)
        gmm1 = GaussianMixture(n_components=1, random_state=42)
        gmm1.fit(speeds_array)
        bic1 = gmm1.bic(speeds_array)
        
        # Fit GMM with 2 components (bimodal)
        gmm2 = GaussianMixture(n_components=2, random_state=42)
        gmm2.fit(speeds_array)
        bic2 = gmm2.bic(speeds_array)
        
        # Calculate BIC difference
        delta_bic = bic1 - bic2
        
        # Bimodal if 2-component model is significantly better
        # Using Raftery (1995) threshold: ΔBIC > 10 = "very strong evidence"
        if delta_bic > 10:
            dist_type = 'bimodal'
        else:
            dist_type = 'unimodal'
    
    results.append({
        'cell_id': cell_id,
        'n_points': n,
        'distribution': dist_type,
        'bic_1component': bic1,
        'bic_2component': bic2,
        'delta_bic': delta_bic
    })

# Convert to dataframe
df_results = pd.DataFrame(results)

# ========== MERGE WITH CELL SUMMARY ==========
print("\nMerging with cell summary data...")

df_cells = pd.read_csv('../data/harvey_cell_summary.csv')
df_results = df_results.merge(df_cells[['cell_id', 'lon_idx', 'lat_idx', 'center_lon', 'center_lat', 
                                          'mean_speed', 'std_speed']], 
                               on='cell_id', how='left')

print(f"✓ Merged with cell coordinates and statistics")

# ========== RESULTS ==========
print("\n" + "="*70)
print("DISTRIBUTION DETECTION RESULTS - HARVEY WRF")
print("="*70)

total_cells = len(df_results)

for dist_type in ['bimodal', 'unimodal', 'too_few_points']:
    count = (df_results['distribution'] == dist_type).sum()
    pct = count / total_cells * 100
    print(f"{dist_type:15s}: {count:4d} cells ({pct:5.1f}%)")

print(f"\nTotal cells analyzed: {total_cells}")

# ========== DETAILED STATISTICS ==========
print("\n" + "="*70)
print("DETAILED STATISTICS")
print("="*70)

# Cells with enough data
cells_with_data = df_results[df_results['n_points'] >= 10]
bimodal_cells = df_results[df_results['distribution'] == 'bimodal']
unimodal_cells = df_results[df_results['distribution'] == 'unimodal']

print(f"\nPoints per cell (all cells):")
print(f"  Mean:   {df_results['n_points'].mean():.1f}")
print(f"  Median: {df_results['n_points'].median():.0f}")
print(f"  Min:    {df_results['n_points'].min()}")
print(f"  Max:    {df_results['n_points'].max()}")

if len(bimodal_cells) > 0:
    print(f"\nBimodal cells characteristics:")
    print(f"  Mean wind speed: {bimodal_cells['mean_speed'].mean():.2f} m/s")
    print(f"  Mean std dev:    {bimodal_cells['std_speed'].mean():.2f} m/s")
    print(f"  Mean ΔBIC:       {bimodal_cells['delta_bic'].mean():.1f}")

if len(unimodal_cells) > 0:
    print(f"\nUnimodal cells characteristics:")
    print(f"  Mean wind speed: {unimodal_cells['mean_speed'].mean():.2f} m/s")
    print(f"  Mean std dev:    {unimodal_cells['std_speed'].mean():.2f} m/s")
    print(f"  Mean ΔBIC:       {unimodal_cells['delta_bic'].mean():.1f}")

# ========== SAVE RESULTS ==========
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

output_file = '../data/harvey_distribution_types_gmm.csv'
df_results.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file}")
print(f"  Columns: {list(df_results.columns)}")

# ========== COMPARISON WITH KATRINA ==========
print("\n" + "="*70)
print("COMPARISON: HARVEY (WRF) vs KATRINA (H*Wind)")
print("="*70)

harvey_bimodal_count = (df_results['distribution'] == 'bimodal').sum()
harvey_unimodal_count = (df_results['distribution'] == 'unimodal').sum()
harvey_few_count = (df_results['distribution'] == 'too_few_points').sum()

print(f"\nHarvey (WRF high-resolution):")
print(f"  Bimodal:        {harvey_bimodal_count:4d} ({harvey_bimodal_count/total_cells*100:5.1f}%)")
print(f"  Unimodal:       {harvey_unimodal_count:4d} ({harvey_unimodal_count/total_cells*100:5.1f}%)")
print(f"  Too few points: {harvey_few_count:4d} ({harvey_few_count/total_cells*100:5.1f}%)")
print(f"  Avg points/cell: {df_results['n_points'].mean():.1f}")

print(f"\nKatrina (H*Wind observations):")
print(f"  Bimodal:        ~30-64 cells (~1-3%)")
print(f"  Unimodal:       ~915-950 cells (~37-38%)")
print(f"  Too few points: ~1521 cells (~61%)")
print(f"  Avg points/cell: ~10.4")

# ========== DATA QUALITY COMPARISON ==========
print("\n" + "="*70)
print("DATA QUALITY INSIGHTS")
print("="*70)

data_density_ratio = df_results['n_points'].mean() / 10.4

print(f"\nData density comparison:")
print(f"  Harvey has {data_density_ratio:.1f}× more points per cell than Katrina")
print(f"  This explains why Harvey shows {'more' if harvey_bimodal_count/total_cells > 0.03 else 'similar or fewer'} bimodal cells")

print("\nPossible reasons for differences:")
print("  1. WRF model data (Harvey) vs. observational data (Katrina)")
print("  2. Different spatial coverage and storm characteristics")
print("  3. WRF provides temporally consistent high-resolution grids")
print("  4. H*Wind depends on sparse observational network")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)


# We selected the Hurricane Harvey WRF 
# (Weather Research and Forecasting)
#  model dataset to address critical 
# data quality limitations observed in
#  the Hurricane Katrina H*Wind observational 
# dataset. The Harvey WRF data provides substantially 
# higher spatial and temporal resolution, with an average
#  of 35.3 data points per grid cell compared to only 10.4 
# for Katrina, and eliminates the data sparsity issue that
#  affected 61% of Katrina's grid cells. This high-resolution 
# numerical weather prediction output ensures complete spatial 
# coverage across all 2,500 grid cells with no missing data, enabling 
# more robust statistical analysis of wind speed distributions.
#  While both datasets exhibit similar proportions of bimodal cells
#  (4.5% for Harvey vs. 1-3% for Katrina), the Harvey dataset's 
# consistent grid structure and dense temporal sampling make it 
# ideal for developing and validating probabilistic wind uncertainty
#  models for UAV path planning applications. The WRF model's 
# physics-based approach also provides physically consistent
#  wind fields that capture the complex dynamics of hurricane 
# environments, which is essential for realistic engagement 
# zone analysis under wind uncertainty.