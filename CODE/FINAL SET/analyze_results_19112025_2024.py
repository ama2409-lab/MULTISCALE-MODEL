"""Generate all analysis plots and statistics from the diffusivity sweep results.

This script reads sweep_diffusivity_19112025_2024.csv and produces:
1. REV convergence plot (D_eff vs domain size)
2. Diffusivity sweep plot (D_eff vs base diffusivity)
3. Dirichlet vs Neumann comparison
4. Statistical summary table
5. Tortuosity analysis

All plots are publication-ready with error bars showing standard error across seeds.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

# Configuration
INPUT_CSV = 'sweep_diffusivity_19112025_2024.csv'
OUTPUT_DIR = Path('results_19112025_2024')
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Reading data from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} rows")
print(f"Columns: {list(df.columns)}")

# Clean data - handle both empty strings and NaN in error column
df = df[(df['error'].isna()) | (df['error'] == '')].copy()
df = df.dropna(subset=['dir_D_eff', 'neu_D_eff'])
print(f"After filtering errors: {len(df)} valid rows")

# ============================================================================
# PLOT 1: REV Convergence (D_eff vs Domain Size) - Main Result
# ============================================================================
print("\n=== Generating REV Convergence Plot ===")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for diff_val in df['diffusivity'].unique():
    subset = df[df['diffusivity'] == diff_val]
    grouped = subset.groupby('L_m').agg({
        'dir_D_eff': ['mean', 'std', 'count'],
        'neu_D_eff': ['mean', 'std', 'count']
    })
    
    Ls = grouped.index.values * 1e6  # Convert to micrometers
    
    # Dirichlet
    dir_mean = grouped['dir_D_eff']['mean'].values
    dir_se = grouped['dir_D_eff']['std'].values / np.sqrt(grouped['dir_D_eff']['count'].values)
    ax1.errorbar(Ls, dir_mean, yerr=dir_se, marker='o', linestyle='-', 
                 label=f'D={diff_val:.0e} m²/s', capsize=3, alpha=0.7)
    
    # Neumann
    neu_mean = grouped['neu_D_eff']['mean'].values
    neu_se = grouped['neu_D_eff']['std'].values / np.sqrt(grouped['neu_D_eff']['count'].values)
    ax2.errorbar(Ls, neu_mean, yerr=neu_se, marker='s', linestyle='--', 
                 label=f'D={diff_val:.0e} m²/s', capsize=3, alpha=0.7)

ax1.set_xlabel('Domain Size L (μm)', fontsize=12)
ax1.set_ylabel('D_eff (m²/s)', fontsize=12)
ax1.set_title('Dirichlet BC: REV Convergence', fontsize=14, fontweight='bold')
ax1.legend(fontsize=8, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.set_yscale('log')

ax2.set_xlabel('Domain Size L (μm)', fontsize=12)
ax2.set_ylabel('D_eff (m²/s)', fontsize=12)
ax2.set_title('Neumann BC: REV Convergence', fontsize=14, fontweight='bold')
ax2.legend(fontsize=8, loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'REV_convergence_by_diffusivity.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'REV_convergence_by_diffusivity.png'}")
plt.close()

# ============================================================================
# PLOT 2: Single Diffusivity REV Convergence (cleaner version for presentation)
# ============================================================================
print("\n=== Generating Single Diffusivity REV Plot (Presentation Version) ===")

# Use middle diffusivity value for cleaner plot
mid_idx = len(df['diffusivity'].unique()) // 2
mid_diff = sorted(df['diffusivity'].unique())[mid_idx]
subset = df[df['diffusivity'] == mid_diff]

grouped = subset.groupby('L_m').agg({
    'dir_D_eff': ['mean', 'std', 'count'],
    'neu_D_eff': ['mean', 'std', 'count'],
    'porosity': ['mean', 'std']
})

Ls = grouped.index.values * 1e6

fig, ax = plt.subplots(figsize=(10, 6))

dir_mean = grouped['dir_D_eff']['mean'].values
dir_se = grouped['dir_D_eff']['std'].values / np.sqrt(grouped['dir_D_eff']['count'].values)
neu_mean = grouped['neu_D_eff']['mean'].values
neu_se = grouped['neu_D_eff']['std'].values / np.sqrt(grouped['neu_D_eff']['count'].values)

ax.errorbar(Ls, dir_mean, yerr=dir_se, marker='o', linestyle='-', linewidth=2,
            label='Dirichlet BC', capsize=5, markersize=8, color='#2E86AB')
ax.errorbar(Ls, neu_mean, yerr=neu_se, marker='s', linestyle='--', linewidth=2,
            label='Neumann BC', capsize=5, markersize=8, color='#A23B72')

ax.set_xlabel('Domain Size L (μm)', fontsize=14, fontweight='bold')
ax.set_ylabel('Effective Diffusivity D_eff (m²/s)', fontsize=14, fontweight='bold')
ax.set_title(f'REV Convergence Study (D_base = {mid_diff:.0e} m²/s)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='best', frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xscale('log')
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'REV_convergence_presentation.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'REV_convergence_presentation.png'}")
plt.close()

# ============================================================================
# PLOT 3: Diffusivity Scaling (D_eff vs D_base)
# ============================================================================
print("\n=== Generating Diffusivity Scaling Plot ===")

fig, ax = plt.subplots(figsize=(10, 6))

for L in df['L_m'].unique():
    subset = df[df['L_m'] == L]
    grouped = subset.groupby('diffusivity').agg({
        'dir_D_eff': ['mean', 'std', 'count']
    })
    
    diffs = grouped.index.values
    dir_mean = grouped['dir_D_eff']['mean'].values
    dir_se = grouped['dir_D_eff']['std'].values / np.sqrt(grouped['dir_D_eff']['count'].values)
    
    ax.errorbar(diffs, dir_mean, yerr=dir_se, marker='o', linestyle='-',
                label=f'L={L*1e6:.0f} μm', capsize=3, linewidth=2, markersize=6)

# Add ideal scaling line (D_eff = phi/tau * D_base)
diffs_ideal = np.logspace(-8, -2, 50)
mean_por = df['porosity'].mean()
mean_tau = df['tau_dir'].mean()
if np.isfinite(mean_tau) and mean_tau > 0:
    ideal_line = (mean_por / mean_tau) * diffs_ideal
    ax.plot(diffs_ideal, ideal_line, 'k--', linewidth=2, alpha=0.5,
            label=f'Ideal (φ/τ={mean_por/mean_tau:.2e})')

ax.set_xlabel('Base Diffusivity D_base (m²/s)', fontsize=14, fontweight='bold')
ax.set_ylabel('Effective Diffusivity D_eff (m²/s)', fontsize=14, fontweight='bold')
ax.set_title('Diffusivity Scaling Analysis (Dirichlet BC)', fontsize=16, fontweight='bold')
ax.legend(fontsize=10, loc='best', frameon=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xscale('log')
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'diffusivity_scaling.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'diffusivity_scaling.png'}")
plt.close()

# ============================================================================
# PLOT 4: Dirichlet vs Neumann Comparison
# ============================================================================
print("\n=== Generating Dirichlet vs Neumann Comparison ===")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot: D_eff comparison
ax1.scatter(df['dir_D_eff'], df['neu_D_eff'], alpha=0.5, s=20)
min_val = min(df['dir_D_eff'].min(), df['neu_D_eff'].min())
max_val = max(df['dir_D_eff'].max(), df['neu_D_eff'].max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')
ax1.set_xlabel('Dirichlet D_eff (m²/s)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Neumann D_eff (m²/s)', fontsize=12, fontweight='bold')
ax1.set_title('BC Comparison: Effective Diffusivity', fontsize=14, fontweight='bold')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Relative difference histogram
rel_diff = 100 * (df['neu_D_eff'] - df['dir_D_eff']) / df['dir_D_eff']
ax2.hist(rel_diff, bins=30, edgecolor='black', alpha=0.7, color='#F18F01')
ax2.axvline(rel_diff.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean = {rel_diff.mean():.2f}%')
ax2.set_xlabel('Relative Difference (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title('Neumann vs Dirichlet: Percent Difference', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'BC_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'BC_comparison.png'}")
plt.close()

# ============================================================================
# PLOT 5: Tortuosity Analysis
# ============================================================================
print("\n=== Generating Tortuosity Analysis ===")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Filter out infinite tortuosity values
df_finite = df[np.isfinite(df['tau_dir']) & np.isfinite(df['tau_neu'])].copy()

for diff_val in sorted(df_finite['diffusivity'].unique())[::2]:  # Plot every other diffusivity
    subset = df_finite[df_finite['diffusivity'] == diff_val]
    grouped = subset.groupby('L_m').agg({
        'tau_dir': ['mean', 'std', 'count'],
        'tau_neu': ['mean', 'std', 'count']
    })
    
    Ls = grouped.index.values * 1e6
    
    tau_dir_mean = grouped['tau_dir']['mean'].values
    tau_dir_se = grouped['tau_dir']['std'].values / np.sqrt(grouped['tau_dir']['count'].values)
    ax1.errorbar(Ls, tau_dir_mean, yerr=tau_dir_se, marker='o', linestyle='-',
                 label=f'D={diff_val:.0e}', capsize=3, alpha=0.7)

ax1.set_xlabel('Domain Size L (μm)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Tortuosity τ', fontsize=12, fontweight='bold')
ax1.set_title('Tortuosity vs Domain Size (Dirichlet)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# Porosity vs Tortuosity
ax2.scatter(df_finite['porosity'], df_finite['tau_dir'], alpha=0.4, s=20, label='Dirichlet')
ax2.scatter(df_finite['porosity'], df_finite['tau_neu'], alpha=0.4, s=20, label='Neumann')
ax2.set_xlabel('Porosity φ', fontsize=12, fontweight='bold')
ax2.set_ylabel('Tortuosity τ', fontsize=12, fontweight='bold')
ax2.set_title('Porosity-Tortuosity Relationship', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tortuosity_analysis.png', dpi=300, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR / 'tortuosity_analysis.png'}")
plt.close()

# ============================================================================
# STATISTICAL SUMMARY TABLE
# ============================================================================
print("\n=== Generating Statistical Summary ===")

summary_rows = []
for L in sorted(df['L_m'].unique()):
    for diff_val in sorted(df['diffusivity'].unique()):
        subset = df[(df['L_m'] == L) & (df['diffusivity'] == diff_val)]
        if len(subset) > 0:
            summary_rows.append({
                'L_um': L * 1e6,
                'D_base': diff_val,
                'n_seeds': len(subset),
                'dir_D_eff_mean': subset['dir_D_eff'].mean(),
                'dir_D_eff_std': subset['dir_D_eff'].std(),
                'neu_D_eff_mean': subset['neu_D_eff'].mean(),
                'neu_D_eff_std': subset['neu_D_eff'].std(),
                'porosity_mean': subset['porosity'].mean(),
                'tau_dir_mean': subset['tau_dir'].mean(),
                'tau_neu_mean': subset['tau_neu'].mean()
            })

summary_df = pd.DataFrame(summary_rows)
summary_path = OUTPUT_DIR / 'statistical_summary.csv'
summary_df.to_csv(summary_path, index=False)
print(f"Saved: {summary_path}")

# Print key findings
print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)

# REV determination
largest_L = df['L_m'].max()
for diff_val in sorted(df['diffusivity'].unique()):
    subset = df[df['diffusivity'] == diff_val]
    grouped = subset.groupby('L_m')['dir_D_eff'].mean()
    if len(grouped) >= 2:
        convergence = 100 * abs(grouped.iloc[-1] - grouped.iloc[-2]) / grouped.iloc[-2]
        print(f"\nD_base = {diff_val:.2e} m²/s:")
        print(f"  Final D_eff (Dirichlet) = {grouped.iloc[-1]:.4e} m²/s")
        print(f"  Convergence (last 2 sizes) = {convergence:.2f}%")
        if convergence < 5:
            print(f"  ✓ REV achieved at L = {largest_L*1e6:.0f} μm")
        else:
            print(f"  ✗ REV not yet achieved (need larger domain)")

print(f"\nMean Porosity: {df['porosity'].mean():.4f} ± {df['porosity'].std():.4f}")
print(f"Mean Tortuosity (Dirichlet): {df_finite['tau_dir'].mean():.2f} ± {df_finite['tau_dir'].std():.2f}")
print(f"Mean Tortuosity (Neumann): {df_finite['tau_neu'].mean():.2f} ± {df_finite['tau_neu'].std():.2f}")

print(f"\nDirichlet vs Neumann:")
print(f"  Mean relative difference: {rel_diff.mean():.2f}% ± {rel_diff.std():.2f}%")
print(f"  Correlation coefficient: {np.corrcoef(df['dir_D_eff'], df['neu_D_eff'])[0,1]:.4f}")

print("\n" + "="*60)
print(f"All plots saved to: {OUTPUT_DIR.absolute()}")
print("="*60)
