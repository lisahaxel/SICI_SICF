#%%
"""
ROC-AUC performance comparison across conditions and methods.
Creates combined violin and box plots for within-subject (SS) and transfer learning (TL) results.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score
from collections import defaultdict


def apply_matplotlib_settings(width_mm=90, height_mm=55):
    """Apply publication-ready matplotlib settings matching fig2b style."""
    mm_to_inch = 1 / 25.4
    
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
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "lines.linewidth": 1.5,
        "lines.markersize": 3,
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "figure.figsize": (width_mm * mm_to_inch, height_mm * mm_to_inch),
    }
    plt.rcParams.update(settings)


def darken_color(hex_color, factor=0.7):
    """Darken a hex color by the given factor."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    darkened_rgb = tuple(int(c * factor) for c in rgb)
    return f"#{darkened_rgb[0]:02x}{darkened_rgb[1]:02x}{darkened_rgb[2]:02x}"


def load_within_subject_results(results_path, condition):
    """Load within-subject cross-validation results."""
    files = list(results_path.glob(f"*_{condition.lower()}_results.pkl"))
    
    subject_scores = []
    
    for file_path in files:
        try:
            with open(file_path, 'rb') as f:
                results = pickle.load(f)
            
            # Extract subject ID from filename
            subject_id = file_path.stem.split('_')[1]
            
            # results structure: [accuracy_arr, model_parameters, final_models, final_features, final_predictions]
            predictions_data = results[4]  # final_predictions
            models_data = results[2]       # final_models
            
            subject_roc_scores = []
            
            # Process each repetition and fold
            for rep_idx, rep_predictions in enumerate(predictions_data):
                for fold_idx, fold_predictions in enumerate(rep_predictions):
                    for model_idx in range(len(fold_predictions)):
                        model_info = models_data[rep_idx][fold_idx][model_idx]
                        y_true = model_info['y_test']
                        y_pred = fold_predictions[model_idx][1]  # test predictions
                        
                        if len(np.unique(y_true)) > 1:  # Ensure both classes present
                            roc_score = roc_auc_score(y_true, y_pred)
                            subject_roc_scores.append(roc_score)
            
            if subject_roc_scores:
                # Take best model performance across folds/repetitions
                best_score = np.max(subject_roc_scores)
                subject_scores.append({'subject_id': subject_id, 'roc_auc': best_score})
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    return pd.DataFrame(subject_scores)


def load_loso_results(results_path, condition):
    """Load LOSO transfer learning results."""
    # Try different possible filenames
    possible_files = [
        results_path / f"LOSO_results_{condition.lower()}.pkl",
        results_path / f"LOSO_results_{condition.lower()}_all_conditions.pkl",
        results_path / f"LOSO_results_{condition.lower()}_standard.pkl"
    ]
    
    loso_file = None
    for file_path in possible_files:
        if file_path.exists():
            loso_file = file_path
            break
    
    if loso_file is None:
        print(f"LOSO file not found for {condition}. Tried: {[f.name for f in possible_files]}")
        return pd.DataFrame(), pd.DataFrame()
    
    try:
        with open(loso_file, 'rb') as f:
            results = pickle.load(f)
        
        print(f"Loaded LOSO file: {loso_file.name}")
        
        zero_shot_data = []
        calibrated_data = []
        
        for subject_id, zero_shot_res in results['zero_shot_results'].items():
            calibrated_res = results['calibrated_results'][subject_id]
            
            # Get best performing model for each subject
            best_zero_shot = max(zero_shot_res.values(), key=lambda x: x.get('accuracy', 0))
            best_calibrated = max(calibrated_res.values(), key=lambda x: x.get('accuracy', 0))
            
            zero_shot_data.append({
                'subject_id': subject_id,
                'roc_auc': best_zero_shot.get('accuracy', 0)
            })
            
            calibrated_data.append({
                'subject_id': subject_id,
                'roc_auc': best_calibrated.get('accuracy', 0)
            })
        
        return pd.DataFrame(zero_shot_data), pd.DataFrame(calibrated_data)
        
    except Exception as e:
        print(f"Error loading LOSO results from {loso_file}: {e}")
        return pd.DataFrame(), pd.DataFrame()


def plot_violin_box(ax, data, position, face_color, edge_color):
    """Plot violin and box at specified position."""
    if len(data) == 0:
        return
    
    # Violin plot - made wider
    parts = ax.violinplot(
        dataset=[data],
        positions=[position],
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.6,  # Increased from 0.4 to 0.6
        bw_method=0.2
    )
    
    for pc in parts['bodies']:
        pc.set_facecolor(face_color)
        pc.set_edgecolor(edge_color)
        pc.set_linewidth(0.5)
        pc.set_alpha(1.0)
        pc.set_zorder(1)
    
    # Box plot - made wider
    bp = ax.boxplot(
        x=[data],
        positions=[position],
        vert=True,
        patch_artist=True,
        widths=0.3,  # Increased from 0.2 to 0.3
        showfliers=False,
        whiskerprops={'color': edge_color, 'linewidth': 0.5, 'zorder': 2},
        capprops={'color': edge_color, 'linewidth': 0.5, 'zorder': 2},
        medianprops={'color': edge_color, 'linewidth': 0.8, 'zorder': 4},
    )
    
    for patch in bp['boxes']:
        patch.set_facecolor('white')
        patch.set_edgecolor(edge_color)
        patch.set_linewidth(0.5)
        patch.set_zorder(3)


def create_performance_plot(data_dict, output_filename="roc_auc_performance.pdf"):
    """Create combined violin and box plot for ROC-AUC performance."""
    
    # Define order: Single, SICI, SICF for both SS and TL
    conditions = ['Single', 'SICI', 'SICF']
    
    # Define colors matching fig2b style
    colors = {
        'SS': '#80a687',           # Green for within-subject
        'Zero-shot': '#9e9e9e',    # Grey for zero-shot
        'Calibrated': '#cee1d1'    # Light green for calibrated
    }
    
    edge_colors = {method: darken_color(color) for method, color in colors.items()}
    
    # Create the plot
    fig, ax = plt.subplots()
    
    # Create positions with proper spacing
    positions = {}
    current_pos = 0
    
    # SS conditions
    for i, condition in enumerate(conditions):
        positions[f'{condition}_SS'] = current_pos
        current_pos += 1.0
    
    current_pos += 0  # Gap between SS and TL
    
    # TL conditions (two bars each)
    for condition in conditions:
        positions[f'{condition}_TL_Zero-shot'] = current_pos
        positions[f'{condition}_TL_Calibrated'] = current_pos + 0.4
        current_pos += 1.5
    
    # Plot all conditions
    for condition in conditions:
        # Within-subject
        ss_key = f'{condition}_SS'
        if ss_key in data_dict and not data_dict[ss_key].empty:
            plot_violin_box(ax, data_dict[ss_key]['roc_auc'], 
                           positions[f'{condition}_SS'],
                           colors['SS'], edge_colors['SS'])
        
        # Transfer learning zero-shot
        zero_key = f'{condition}_TL_zero'
        if zero_key in data_dict and not data_dict[zero_key].empty:
            plot_violin_box(ax, data_dict[zero_key]['roc_auc'],
                           positions[f'{condition}_TL_Zero-shot'],
                           colors['Zero-shot'], edge_colors['Zero-shot'])
        
        # Transfer learning calibrated
        cal_key = f'{condition}_TL_cal'
        if cal_key in data_dict and not data_dict[cal_key].empty:
            plot_violin_box(ax, data_dict[cal_key]['roc_auc'],
                           positions[f'{condition}_TL_Calibrated'],
                           colors['Calibrated'], edge_colors['Calibrated'])
    
    # Set x-axis labels
    x_labels = ['Single-SS', 'SICI-SS', 'SICF-SS', 'Single-TL', 'SICI-TL', 'SICF-TL']
    tick_positions = [0, 1, 2, 3.2, 4.7, 6.2]  # Centered positions for each group
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    
    # Styling
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.set_ylabel('ROC-AUC')
    ax.set_ylim(0.4, 0.9)
    
    # Create legend in upper right
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['SS'], label='Within-subject'),
        Patch(facecolor=colors['Zero-shot'], label='Zero-shot'),
        Patch(facecolor=colors['Calibrated'], label='Calibrated')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.savefig(output_filename.replace('.pdf', '.png'), dpi=300)
    print(f"Plot saved to: {output_filename}")

def print_performance_summary(data_dict):
    """Print mean ± std ROC-AUC for each condition."""
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY (ROC-AUC)")
    print("="*50)
    
    conditions = ['Single', 'SICI', 'SICF']
    
    for condition in conditions:
        print(f"\n{condition}:")
        
        # Within-subject
        ss_key = f'{condition}_SS'
        if ss_key in data_dict and not data_dict[ss_key].empty:
            ss_scores = data_dict[ss_key]['roc_auc']
            ss_mean = ss_scores.mean()
            ss_std = ss_scores.std()
            print(f"  Within-subject: {ss_mean:.3f} ± {ss_std:.3f} (n={len(ss_scores)})")
        else:
            print(f"  Within-subject: No data")
        
        # Zero-shot
        zero_key = f'{condition}_TL_zero'
        if zero_key in data_dict and not data_dict[zero_key].empty:
            zero_scores = data_dict[zero_key]['roc_auc']
            zero_mean = zero_scores.mean()
            zero_std = zero_scores.std()
            print(f"  Zero-shot:      {zero_mean:.3f} ± {zero_std:.3f} (n={len(zero_scores)})")
        else:
            print(f"  Zero-shot:      No data")
        
        # Calibrated
        cal_key = f'{condition}_TL_cal'
        if cal_key in data_dict and not data_dict[cal_key].empty:
            cal_scores = data_dict[cal_key]['roc_auc']
            cal_mean = cal_scores.mean()
            cal_std = cal_scores.std()
            print(f"  Calibrated:     {cal_mean:.3f} ± {cal_std:.3f} (n={len(cal_scores)})")
        else:
            print(f"  Calibrated:     No data")
    
    print("\n" + "="*50)


def main():
    """Main execution function."""
    # Configuration
    results_path = Path("/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/SICISICF/results/models")
    
    if not results_path.exists():
        print(f"Results path does not exist: {results_path}")
        return
    
    apply_matplotlib_settings()
    
    # Load all data in the desired order: Single, SICI, SICF
    data_dict = {}
    conditions = ['Single', 'SICI', 'SICF']
    
    print("Loading performance data...")
    
    for condition in conditions:
        print(f"Processing {condition}...")
        
        # Load within-subject results
        ss_data = load_within_subject_results(results_path, condition)
        data_dict[f'{condition}_SS'] = ss_data
        print(f"  Within-subject: {len(ss_data)} subjects")
        
        # Load LOSO results
        zero_shot_data, calibrated_data = load_loso_results(results_path, condition)
        data_dict[f'{condition}_TL_zero'] = zero_shot_data
        data_dict[f'{condition}_TL_cal'] = calibrated_data
        print(f"  Zero-shot: {len(zero_shot_data)} subjects")
        print(f"  Calibrated: {len(calibrated_data)} subjects")
    
    # Create the plot
    create_performance_plot(data_dict)

    # Print performance summary
    print_performance_summary(data_dict)


if __name__ == "__main__":
    main()
# %%
