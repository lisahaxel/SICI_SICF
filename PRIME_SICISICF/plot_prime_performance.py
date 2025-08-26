# %%
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

def apply_matplotlib_settings():
    """Applies the specified matplotlib settings for the plot."""
    settings = {
        "text.usetex": False,
        "mathtext.default": "regular",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
        "font.size": 7,
        "figure.titlesize": 7,
        "legend.fontsize": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "image.interpolation": "nearest",
        "image.resample": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "lines.linewidth": 1.5,
        "lines.markersize": 3,
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "savefig.format": "svg",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "svg.fonttype": "none",
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
    plt.rcParams.update(settings)

def darken_color(hex_color, factor=0.7):
    """Darken a hex color by the given factor"""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    darkened_rgb = tuple(int(c * factor) for c in rgb)
    return f"#{darkened_rgb[0]:02x}{darkened_rgb[1]:02x}{darkened_rgb[2]:02x}"

def perform_statistical_tests_calibration(data_df):
    """
    Performs one-sided paired t-tests with Benjamini-Hochberg correction for calibration data.
    Tests if PRE-FT is significantly better than other conditions.
    """
    if data_df is None or data_df.empty:
        return {}
    
    # Get unique conditions
    conditions = data_df['condition'].unique()
    ref_condition = 'PRE-FT'
    
    if ref_condition not in conditions:
        print(f"Warning: Reference condition '{ref_condition}' not found in data")
        return {}
    
    # Get reference data
    ref_data_all = data_df[(data_df['condition'] == ref_condition) & (data_df['variant'] == 'All')]['roc_auc'].values
    ref_data_extreme = data_df[(data_df['condition'] == ref_condition) & (data_df['variant'] == 'Extreme')]['roc_auc'].values
    
    # Perform tests for each condition
    p_values = []
    test_results = {}
    
    for condition in conditions:
        if condition == ref_condition:
            test_results[condition] = {'all': 0, 'extreme': 0}  # No stars for reference
            continue
            
        cond_data_all = data_df[(data_df['condition'] == condition) & (data_df['variant'] == 'All')]['roc_auc'].values
        cond_data_extreme = data_df[(data_df['condition'] == condition) & (data_df['variant'] == 'Extreme')]['roc_auc'].values
        
        # Ensure we have paired data (same subjects)
        if len(cond_data_all) != len(ref_data_all) or len(cond_data_extreme) != len(ref_data_extreme):
            print(f"Warning: Sample sizes don't match for {condition}. Using available data.")
            min_len_all = min(len(cond_data_all), len(ref_data_all))
            min_len_extreme = min(len(cond_data_extreme), len(ref_data_extreme))
            cond_data_all = cond_data_all[:min_len_all]
            ref_data_all_paired = ref_data_all[:min_len_all]
            cond_data_extreme = cond_data_extreme[:min_len_extreme]
            ref_data_extreme_paired = ref_data_extreme[:min_len_extreme]
        else:
            ref_data_all_paired = ref_data_all
            ref_data_extreme_paired = ref_data_extreme
        
        # Perform one-sided paired t-tests (PRE-FT should be better)
        _, p_all = stats.ttest_rel(ref_data_all_paired, cond_data_all, alternative='greater')
        _, p_extreme = stats.ttest_rel(ref_data_extreme_paired, cond_data_extreme, alternative='greater')
        
        p_values.extend([p_all, p_extreme])
        test_results[condition] = {'all': p_all, 'extreme': p_extreme}
    
    # Apply Benjamini-Hochberg correction
    if p_values:
        rejected, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
        
        # Map corrected p-values back to conditions
        p_idx = 0
        for condition in conditions:
            if condition == ref_condition:
                continue
            test_results[condition]['all_corrected'] = p_corrected[p_idx]
            test_results[condition]['extreme_corrected'] = p_corrected[p_idx + 1]
            p_idx += 2
    
    return test_results

def get_significance_stars(p_value):
    """Convert p-value to significance stars."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''

def plot_calibration_violins(ax, csv_path_pretrain, csv_path_no_pretrain):
    """Creates grouped violin plots for calibration and fine-tuning results from two sources."""
    
    # 1. Load and process standard (pretrained) data
    try:
        df_pretrain = pd.read_csv(csv_path_pretrain)
        print(f"Pretrained CSV columns: {list(df_pretrain.columns)}")
    except FileNotFoundError:
        print(f"Error: Pretrained CSV file not found at {csv_path_pretrain}")
        ax.text(0.5, 0.5, 'Pretrained CSV not found.', ha='center', va='center')
        return
    
    # Define the columns we want from pretrained data (including pre-calibration)
    value_vars_pre = [
        'pre_calib_zero_shot_roc_auc_all', 'post_calib_zero_shot_roc_auc_all', 'finetuned_roc_auc_all',
        'pre_calib_zero_shot_roc_auc_extreme', 'post_calib_zero_shot_roc_auc_extreme', 'finetuned_roc_auc_extreme'
    ]
    
    # Check which columns actually exist
    available_pre_cols = [col for col in value_vars_pre if col in df_pretrain.columns]
    missing_pre_cols = [col for col in value_vars_pre if col not in df_pretrain.columns]
    
    if missing_pre_cols:
        print(f"Warning: Missing columns in pretrained data: {missing_pre_cols}")
    
    if not available_pre_cols:
        print("Error: No matching columns found in pretrained data")
        return
    
    # Melt the pretrained data
    df_long_pre = df_pretrain.melt(id_vars=['subject_id'], value_vars=available_pre_cols, 
                                   var_name='metric', value_name='roc_auc')
    
    # Process pretrained data
    df_long_pre['variant'] = np.where(df_long_pre['metric'].str.contains('_extreme'), 'Extreme', 'All')
    
    # Clean up condition names for pretrained data
    df_long_pre['condition'] = df_long_pre['metric'].copy()
    df_long_pre['condition'] = df_long_pre['condition'].str.replace('_zero_shot_roc_auc_extreme', '')
    df_long_pre['condition'] = df_long_pre['condition'].str.replace('_zero_shot_roc_auc_all', '')
    df_long_pre['condition'] = df_long_pre['condition'].str.replace('_roc_auc_extreme', '')
    df_long_pre['condition'] = df_long_pre['condition'].str.replace('_roc_auc_all', '')
    
    # Map to display names
    df_long_pre['condition'] = df_long_pre['condition'].replace({
        'pre_calib': 'PRE-ZS',
        'post_calib': 'PRE-CAL', 
        'finetuned': 'PRE-FT'
    })
    
    print(f"Pretrained conditions found: {df_long_pre['condition'].unique()}")

    # 2. Load and process "no pretrain" (from scratch) data
    try:
        df_no_pretrain = pd.read_csv(csv_path_no_pretrain)
        print(f"No-pretrain CSV columns: {list(df_no_pretrain.columns)}")
    except FileNotFoundError:
        print(f"Error: 'No Pretrain' CSV file not found at {csv_path_no_pretrain}")
        df_combined = df_long_pre
    else:
        # Define columns for no-pretrain data
        value_vars_no_pre = [
            'post_calib_zero_shot_roc_auc_all', 'finetuned_roc_auc_all',
            'post_calib_zero_shot_roc_auc_extreme', 'finetuned_roc_auc_extreme'
        ]
        
        # Check which columns actually exist
        available_no_pre_cols = [col for col in value_vars_no_pre if col in df_no_pretrain.columns]
        missing_no_pre_cols = [col for col in value_vars_no_pre if col not in df_no_pretrain.columns]
        
        if missing_no_pre_cols:
            print(f"Warning: Missing columns in no-pretrain data: {missing_no_pre_cols}")
        
        if available_no_pre_cols:
            # Melt the no-pretrain data
            df_long_no_pre = df_no_pretrain.melt(id_vars=['subject_id'], value_vars=available_no_pre_cols, 
                                               var_name='metric', value_name='roc_auc')
            
            # Process no-pretrain data
            df_long_no_pre['variant'] = np.where(df_long_no_pre['metric'].str.contains('_extreme'), 'Extreme', 'All')
            
            # Clean up condition names for no-pretrain data
            df_long_no_pre['condition'] = df_long_no_pre['metric'].copy()
            df_long_no_pre['condition'] = df_long_no_pre['condition'].str.replace('_zero_shot_roc_auc_extreme', '')
            df_long_no_pre['condition'] = df_long_no_pre['condition'].str.replace('_zero_shot_roc_auc_all', '')
            df_long_no_pre['condition'] = df_long_no_pre['condition'].str.replace('_roc_auc_extreme', '')
            df_long_no_pre['condition'] = df_long_no_pre['condition'].str.replace('_roc_auc_all', '')
            
            # Map to display names
            df_long_no_pre['condition'] = df_long_no_pre['condition'].replace({
                'post_calib': 'SS-CAL',
                'finetuned': 'SS-FT'
            })
            
            print(f"No-pretrain conditions found: {df_long_no_pre['condition'].unique()}")
            
            # Combine the datasets
            df_combined = pd.concat([df_long_pre, df_long_no_pre], ignore_index=True)
        else:
            print("No valid columns found in no-pretrain data, using only pretrained data")
            df_combined = df_long_pre

    # 3. Define colors and order based on what we actually have
    all_conditions = df_combined['condition'].unique()
    condition_order = []
    
    # Add conditions in preferred order if they exist
    preferred_order = ['SS-CAL', 'SS-FT', 'PRE-ZS', 'PRE-CAL', 'PRE-FT']
    for cond in preferred_order:
        if cond in all_conditions:
            condition_order.append(cond)
    
    print(f"Final condition order: {condition_order}")
    
    colors_all = {
        'SS-CAL': '#9e9e9e', 'SS-FT': '#9e9e9e', 'PRE-ZS': '#9e9e9e',
        'PRE-CAL': '#9e9e9e', 'PRE-FT': '#80a687'
    }
    colors_extreme = {
        'SS-CAL': '#e1e1e0', 'SS-FT': '#e1e1e0', 'PRE-ZS': '#e1e1e0',
        'PRE-CAL': '#e1e1e0', 'PRE-FT': '#cee1d1'
    }

    # 4. Create interaction variable and custom positions for closer All/Extreme pairs
    df_combined['cat_variant'] = df_combined['condition'].astype(str) + '_' + df_combined['variant']
    plot_order = [f'{c}_{v}' for c in condition_order for v in ['All', 'Extreme']]

    # Create custom positions - closer pairs, wider gaps between conditions
    position_mapping = {}
    current_pos = 0
    for i, c in enumerate(condition_order):
        # All and Extreme close together
        position_mapping[f'{c}_All'] = current_pos
        position_mapping[f'{c}_Extreme'] = current_pos + 0.4  # Small gap within pair
        current_pos += 1.5  # Larger gap between condition pairs

    # Create palettes that map the interaction term to colors
    palette_dict = {}
    edge_palette_dict = {}
    for c in condition_order:
        palette_dict[f'{c}_All'] = colors_all.get(c, '#333333')
        palette_dict[f'{c}_Extreme'] = colors_extreme.get(c, '#CCCCCC')
        edge_palette_dict[f'{c}_All'] = darken_color(colors_all.get(c, '#333333'))
        edge_palette_dict[f'{c}_Extreme'] = darken_color(colors_extreme.get(c, '#CCCCCC'))

    # 5. Perform statistical tests
    stat_results = perform_statistical_tests_calibration(df_combined)

    # 6. Layer 1: Violin Plot with custom positions
    for cat_var in plot_order:
        data_subset = df_combined[df_combined['cat_variant'] == cat_var]
        if data_subset.empty:
            print(f"Warning: No data for {cat_var}")
            continue
        
        pos = position_mapping[cat_var]
        color = palette_dict.get(cat_var, '#333333')
        edge_color = edge_palette_dict.get(cat_var, '#333333')
        
        # Check if we have enough data points for violin plot
        roc_data = data_subset['roc_auc'].dropna()
        if len(roc_data) < 2:
            print(f"Warning: Not enough data points for violin plot of {cat_var} (n={len(roc_data)})")
            continue
            
        parts = ax.violinplot(
            dataset=roc_data,
            positions=[pos],
            showmeans=False,
            showmedians=False,
            showextrema=False,
            widths=0.4,
            bw_method=0.2
        )
        
        # Style the violin
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor(edge_color)
            pc.set_linewidth(0.5)
            pc.set_alpha(1.0)
            pc.set_zorder(1)

    # 7. Layer 2: Box Plot with custom positions
    for cat_var in plot_order:
        data_subset = df_combined[df_combined['cat_variant'] == cat_var]
        box_data = data_subset['roc_auc'].dropna()
        if box_data.empty:
            continue

        box_color = palette_dict.get(cat_var, 'gray')
        edge_color = edge_palette_dict.get(cat_var, 'gray')
        box_pos = position_mapping[cat_var]

        bp = ax.boxplot(
            x=box_data,
            positions=[box_pos],
            vert=True,
            patch_artist=True,
            widths=0.2,
            showfliers=False,
            whiskerprops={'color': edge_color, 'linewidth': 0.5, 'zorder': 2},
            capprops={'color': edge_color, 'linewidth': 0.5, 'zorder': 2},
            medianprops={'color': edge_color, 'linewidth': 0.8, 'zorder': 4},
        )
        for patch in bp['boxes']:
            patch.set_facecolor('w')
            patch.set_edgecolor(edge_color)
            patch.set_linewidth(0.5)
            patch.set_zorder(3)

    # 8. Add significance stars with custom positions
    y_max = ax.get_ylim()[1]
    star_offset = 0.02
    
    for cat_var in plot_order:
        condition = cat_var.split('_')[0]
        # Handle multi-word conditions like 'SS-CAL' or 'PRE-FT'
        if len(cat_var.split('_')) > 2:
            condition = '_'.join(cat_var.split('_')[:-1])
        
        variant = cat_var.split('_')[-1]
        pos = position_mapping.get(cat_var)
        
        if pos is None:
            continue
            
        if condition in stat_results:
            if variant == 'All' and 'all_corrected' in stat_results[condition]:
                stars = get_significance_stars(stat_results[condition]['all_corrected'])
            elif variant == 'Extreme' and 'extreme_corrected' in stat_results[condition]:
                stars = get_significance_stars(stat_results[condition]['extreme_corrected'])
            else:
                stars = ''
            
            if stars:
                ax.text(pos, y_max - star_offset, stars, 
                       ha='center', va='top', fontsize=6, fontweight='bold',
                       zorder=5)

    # 9. Customize the x-axis labels with custom positions
    if condition_order:
        tick_positions = [i * 1.5 + 0.2 for i in range(len(condition_order))]  # Center of each pair
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(condition_order, ha="right")
        ax.tick_params(axis='x', length=3)
    ax.set_xlabel('')

    # 10. Create legend - horizontal at bottom inside plot
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#555555', label='All'),
        Patch(facecolor='#CCCCCC', label='Extreme')
    ]
    ax.legend(handles=legend_elements, 
             loc='lower right',
             ncol=2,  # Put legend items in one row
             handlelength=1,
             columnspacing=1.0,
             frameon=False)

    # 11. Add chance level line and finalize
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1)
    ax.set_ylabel('ROC-AUC')


    # Print statistical results
    print("Statistical test results for calibration:")
    for condition, results in stat_results.items():
        if 'all_corrected' in results:
            print(f"  {condition}: All p={results['all_corrected']:.4f}, Extreme p={results['extreme_corrected']:.4f}")
    
    # Print data summary
    print(f"\nData summary:")
    print(f"Total data points: {len(df_combined)}")
    for condition in condition_order:
        for variant in ['All', 'Extreme']:
            count = len(df_combined[(df_combined['condition'] == condition) & (df_combined['variant'] == variant)])
            print(f"  {condition} {variant}: {count} data points")



# --- helper -----------------------------------------------------------
def to_binary(y, thresh=0.5):
    """Convert continuous labels to 0/1, returning None if only one class is present."""
    y_bin = (y >= thresh).astype(int)
    if y_bin.min() == y_bin.max():
        return None
    return y_bin

def main():
    """Main function to generate and save the plots."""
    # for SICF
    PRIME_PATH = "/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/SICISICF/PRIME_SICISICF/results_prime/10ms_pp_w50/AlignEval_PRIME_SICFClassification_FM-Full_A-None_AdaBN-F"
    CSV_PATH_PRETRAIN = "/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/SICISICF/PRIME_SICISICF/results_prime/10ms_pp_w50/AlignEval_PRIME_SICFClassification_FM-DecThr_A-None_AdaBN-F/AlignEval_PRIME_SICFClassification_FM-DecThr_A-None_AdaBN-F/20250805_141904/results_summary.csv"
    CSV_PATH_NO_PRETRAIN = "/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/SICISICF/PRIME_SICISICF/results_prime/10ms_pp_w50/AlignEval_PRIME_SICFClassification_FM-Full_A-None_AdaBN-F_no_pretrain/AlignEval_PRIME_SICFClassification_FM-Full_A-None_AdaBN-F_no_pretrain/20250805_142843/results_summary.csv"

    # # for SICI
    PRIME_PATH = "/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/SICISICF/PRIME_SICISICF/results_prime/10ms_pp_w50/AlignEval_PRIME_SICIClassification_FM-Full_A-None_AdaBN-F"
    CSV_PATH_PRETRAIN = "/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/SICISICF/PRIME_SICISICF/results_prime/10ms_pp_w50/AlignEval_PRIME_SICIClassification_FM-Full_A-None_AdaBN-F/AlignEval_PRIME_SICIClassification_FM-Full_A-None_AdaBN-F/20250805_141854/results_summary.csv"
    CSV_PATH_NO_PRETRAIN = "/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/SICISICF/PRIME_SICISICF/results_prime/10ms_pp_w50/AlignEval_PRIME_SICIClassification_FM-Full_A-None_AdaBN-F_no_pretrain/AlignEval_PRIME_SICIClassification_FM-Full_A-None_AdaBN-F_no_pretrain/20250805_142851/results_summary.csv"
    
    # # for Single Pulse
    PRIME_PATH = "/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/SICISICF/PRIME_SICISICF/results_prime/10ms_pp_w50/AlignEval_PRIME_SinglePulseClassification_FM-Full_A-None_AdaBN-F"
    CSV_PATH_PRETRAIN = "/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/SICISICF/PRIME_SICISICF/results_prime/10ms_pp_w50/AlignEval_PRIME_SinglePulseClassification_FM-Full_A-None_AdaBN-F/AlignEval_PRIME_SinglePulseClassification_FM-Full_A-None_AdaBN-F/20250805_141856/results_summary.csv"
    CSV_PATH_NO_PRETRAIN = "/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/SICISICF/PRIME_SICISICF/results_prime/10ms_pp_w50/AlignEval_PRIME_SICIClassification_FM-Full_A-None_AdaBN-F_no_pretrain/AlignEval_PRIME_SICIClassification_FM-Full_A-None_AdaBN-F_no_pretrain/20250805_143331/results_summary.csv"




    apply_matplotlib_settings()


    # --- Plot: Calibration and Fine-Tuning Violins ---
    fig2, ax2 = plt.subplots(figsize=(2.8, 1.25))
    plot_calibration_violins(ax2, CSV_PATH_PRETRAIN, CSV_PATH_NO_PRETRAIN)
    fig2.tight_layout(pad=0.5)
    output_filename2 = "SinglePulse_calibration_violins.png"
    plt.savefig(output_filename2, dpi=300)
    print(f"\nViolin plot saved successfully to {output_filename2}")


    plt.show()

if __name__ == "__main__":
    main()

# %%