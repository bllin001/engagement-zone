import pandas as pd
import numpy as np
from collections import defaultdict

print("="*70)
print("CALCULATE WIND PROBABILITIES FOR EZ CALCULATION")
print("="*70)

# ========== LOAD DATA ==========
print("\nLoading Harvey data...")

df_cells = pd.read_csv('../data/harvey_cell_summary.csv')
df_dist = pd.read_csv('../data/harvey_distribution_types_gmm.csv')

# Merge
df = df_cells.merge(df_dist[['cell_id', 'distribution']], 
                    on='cell_id', how='left')

# Load raw data for bimodal calculations
df_points = pd.read_csv('../data/harvey_all_points_with_cells.csv')

# Group speeds by cell
cell_speeds = defaultdict(list)
for _, row in df_points.iterrows():
    cell_speeds[row['cell_id']].append(row['WIND_SPEED'])

print(f"✓ Loaded {len(df):,} cells")
print(f"✓ Loaded {len(df_points):,} wind measurements")

# ========== CALCULATE PROBABILITIES ==========
print("\nCalculating wind probabilities...")

results = []

for _, row in df.iterrows():
    cell_id = row['cell_id']
    dist_type = row['distribution']
    
    speeds = np.array(cell_speeds[cell_id])
    
    if len(speeds) < 2:
        # Insufficient data
        results.append({
            'cell_id': cell_id,
            'lon_idx': row['lon_idx'],
            'lat_idx': row['lat_idx'],
            'center_lon': row['center_lon'],
            'center_lat': row['center_lat'],
            'distribution': dist_type,
            'n_regimes': 0,
            'wind_speed_1': np.nan,
            'probability_1': np.nan,
            'std_1': np.nan,
            'wind_speed_2': np.nan,
            'probability_2': np.nan,
            'std_2': np.nan
        })
        continue
    
    if dist_type == 'bimodal':
        # ========== BIMODAL: TWO WIND SCENARIOS ==========
        
        # Split at mean
        mean_val = np.mean(speeds)
        high_mask = speeds > mean_val
        low_mask = speeds <= mean_val
        
        n_high = np.sum(high_mask)
        n_low = np.sum(low_mask)
        
        if n_high > 0 and n_low > 0:
            # Regime 1 (Lower winds)
            wind_speed_1 = np.mean(speeds[low_mask])
            probability_1 = n_low / len(speeds)
            std_1 = np.std(speeds[low_mask], ddof=0) if n_low > 1 else 0.0
            
            # Regime 2 (Higher winds)
            wind_speed_2 = np.mean(speeds[high_mask])
            probability_2 = n_high / len(speeds)
            std_2 = np.std(speeds[high_mask], ddof=0) if n_high > 1 else 0.0
            
            n_regimes = 2
        else:
            # Fallback to unimodal
            wind_speed_1 = row['mean_speed']
            probability_1 = 1.0
            std_1 = row['std_speed']
            wind_speed_2 = np.nan
            probability_2 = np.nan
            std_2 = np.nan
            n_regimes = 1
    
    else:
        # ========== UNIMODAL: ONE WIND SCENARIO ==========
        
        wind_speed_1 = row['mean_speed']
        probability_1 = 1
        std_1 = row['std_speed']
        wind_speed_2 = np.nan
        probability_2 = np.nan
        std_2 = np.nan
        n_regimes = 1
    
    # Store results
    results.append({
        'cell_id': cell_id,
        'lon_idx': row['lon_idx'],
        'lat_idx': row['lat_idx'],
        'center_lon': row['center_lon'],
        'center_lat': row['center_lat'],
        'distribution': dist_type,
        'n_regimes': n_regimes,
        'wind_speed_1': wind_speed_1,
        'probability_1': probability_1,
        'std_1': std_1,
        'wind_speed_2': wind_speed_2,
        'probability_2': probability_2,
        'std_2': std_2
    })

# Convert to DataFrame
df_probs = pd.DataFrame(results)

# ========== SAVE RESULTS ==========
output_file = '../output/harvey_wind_probabilities_for_EZ.csv'
df_probs.to_csv(output_file, index=False)
print(f"\n✓ Saved: {output_file}")

# ========== STATISTICS ==========
print("\n" + "="*70)
print("WIND PROBABILITY STATISTICS")
print("="*70)

unimodal_cells = df_probs[df_probs['n_regimes'] == 1]
bimodal_cells = df_probs[df_probs['n_regimes'] == 2]

print(f"\nTotal cells: {len(df_probs):,}")
print(f"  Unimodal (1 regime):  {len(unimodal_cells):,} ({len(unimodal_cells)/len(df_probs)*100:.1f}%)")
print(f"  Bimodal (2 regimes):  {len(bimodal_cells):,} ({len(bimodal_cells)/len(df_probs)*100:.1f}%)")

print(f"\n" + "="*70)
print("UNIMODAL CELLS")
print("="*70)
print(f"Wind speed (μ):")
print(f"  Mean: {unimodal_cells['wind_speed_1'].mean():.2f} m/s")
print(f"  Range: {unimodal_cells['wind_speed_1'].min():.2f} - {unimodal_cells['wind_speed_1'].max():.2f} m/s")
print(f"Probability: p = 1.0 (always)")

if len(bimodal_cells) > 0:
    print(f"\n" + "="*70)
    print("BIMODAL CELLS")
    print("="*70)
    
    print(f"\nRegime 1 (Lower winds):")
    print(f"  Mean wind speed: {bimodal_cells['wind_speed_1'].mean():.2f} m/s")
    print(f"  Range: {bimodal_cells['wind_speed_1'].min():.2f} - {bimodal_cells['wind_speed_1'].max():.2f} m/s")
    print(f"  Mean probability: {bimodal_cells['probability_1'].mean():.3f}")
    print(f"  Probability range: {bimodal_cells['probability_1'].min():.3f} - {bimodal_cells['probability_1'].max():.3f}")
    
    print(f"\nRegime 2 (Higher winds):")
    print(f"  Mean wind speed: {bimodal_cells['wind_speed_2'].mean():.2f} m/s")
    print(f"  Range: {bimodal_cells['wind_speed_2'].min():.2f} - {bimodal_cells['wind_speed_2'].max():.2f} m/s")
    print(f"  Mean probability: {bimodal_cells['probability_2'].mean():.3f}")
    print(f"  Probability range: {bimodal_cells['probability_2'].min():.3f} - {bimodal_cells['probability_2'].max():.3f}")
    
    print(f"\nProbability verification:")
    print(f"  p₁ + p₂ = {bimodal_cells['probability_1'].mean() + bimodal_cells['probability_2'].mean():.3f} (should be 1.0)")

# ========== EXAMPLE CELLS ==========
print(f"\n" + "="*70)
print("EXAMPLE CELLS")
print("="*70)

# Example unimodal
uni_example = unimodal_cells.iloc[0]
print(f"\nExample UNIMODAL Cell (ID={uni_example['cell_id']}):")
print(f"  Distribution: {uni_example['distribution']}")
print(f"  Regime 1: w = {uni_example['wind_speed_1']:.2f} m/s, p = {uni_example['probability_1']:.3f}")
print(f"  EZ calculation: EZ = 1.0 × EZ(w={uni_example['wind_speed_1']:.2f})")

# Example bimodal
if len(bimodal_cells) > 0:
    bi_example = bimodal_cells.iloc[0]
    print(f"\nExample BIMODAL Cell (ID={bi_example['cell_id']}):")
    print(f"  Distribution: {bi_example['distribution']}")
    print(f"  Regime 1: w = {bi_example['wind_speed_1']:.2f} m/s, p = {bi_example['probability_1']:.3f}")
    print(f"  Regime 2: w = {bi_example['wind_speed_2']:.2f} m/s, p = {bi_example['probability_2']:.3f}")
    print(f"  EZ calculation: EZ = {bi_example['probability_1']:.3f} × EZ(w={bi_example['wind_speed_1']:.2f}) + {bi_example['probability_2']:.3f} × EZ(w={bi_example['wind_speed_2']:.2f})")

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print(f"\nOutput file: {output_file}")
print("Columns:")
print("  - cell_id, lon_idx, lat_idx, center_lon, center_lat")
print("  - distribution: 'unimodal' or 'bimodal'")
print("  - n_regimes: 1 or 2")
print("  - wind_speed_1, probability_1, std_1: Regime 1 (or only regime)")
print("  - wind_speed_2, probability_2, std_2: Regime 2 (NaN for unimodal)")
print("\nUse this for EZ calculation:")
print("  - Unimodal: EZ = EZ(wind_speed_1)")
print("  - Bimodal:  EZ = probability_1 × EZ(wind_speed_1) + probability_2 × EZ(wind_speed_2)")