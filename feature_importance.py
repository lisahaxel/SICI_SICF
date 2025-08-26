#%%
"""
EEG Feature Distinctiveness Analysis
Analyzes whether SICI, SICF, and Single conditions rely on overlapping or distinct predictive features.
"""

import os
import re
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_matplotlib_settings():
    """Apply consistent matplotlib settings matching the style from fig3.py"""
    settings = {
        "text.usetex": False,
        "mathtext.default": "regular",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
        "font.size": 6,
        "figure.titlesize": 6,
        "legend.fontsize": 6,
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
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
        "svg.fonttype": "none",
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
    plt.rcParams.update(settings)

def darken_color(hex_color, factor=0.7):
    """Darken a hex color by a given factor."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    darkened_rgb = tuple(int(c * factor) for c in rgb)
    return f"#{darkened_rgb[0]:02x}{darkened_rgb[1]:02x}{darkened_rgb[2]:02x}"

def get_significance_stars(p_value):
    """Convert p-value to significance stars."""
    if p_value < 0.001: return '***'
    if p_value < 0.01: return '**'
    if p_value < 0.05: return '*'
    return ''

class EEGPermutationAnalyzer:
    """Analyzes EEG features using permutation importance from pre-trained models."""

    def __init__(self, results_path, n_jobs=-1):
        self.results_path = Path(results_path)
        self.conditions = ['Single', 'SICI', 'SICF']
        self.n_jobs = n_jobs
        self.all_feature_importances = {}
        self._data_cache = {}
    
    def get_available_subjects(self):
        """Find all unique subject IDs from the results filenames."""
        subjects = set()
        for file in self.results_path.glob("Sub_*_results.pkl"):
            for part in file.stem.split('_'):
                if part.startswith('sub-'):
                    subjects.add(part)
                    break
        return sorted(list(subjects))

    def load_subject_data(self, subject_id, condition):
        """Load pre-computed results for a subject and condition."""
        cache_key = (subject_id, condition)
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        pattern = f"Sub_*_{subject_id}_{condition.lower()}_results.pkl"
        try:
            filepath = next(self.results_path.glob(pattern))
            with open(filepath, 'rb') as f:
                all_results = pickle.load(f)
                final_models = all_results[2] if len(all_results) > 2 else None
                self._data_cache[cache_key] = final_models
                return final_models
        except (StopIteration, FileNotFoundError):
            logger.warning(f"Results file not found for {subject_id} - {condition}")
            return None

    def _compute_permutation_importance_single_model(self, model_info):
        """Compute permutation importance for a single model using ROC-AUC."""
        try:
            model = model_info['model']
            # Load the scaler that was saved along with the model
            scaler = model_info['scaler']
            X_test = model_info['X_test']
            y_test = model_info['y_test']
            features_for_fold = model_info['features']

            # Select the relevant feature columns from the raw test data
            X_test_subset = X_test[features_for_fold]
            
            # Apply the same scaling that was used during training
            X_test_scaled = scaler.transform(X_test_subset)

            y_test_array = y_test.values if hasattr(y_test, 'values') else y_test

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

                perm_result = permutation_importance(
                    model, X_test_scaled, y_test_array,  # Use the scaled data here
                    n_repeats=10, random_state=42, n_jobs=-1,
                    scoring='roc_auc'
                )

            return dict(zip(features_for_fold, perm_result.importances_mean))

        except Exception as e:
            logger.error(f"Error computing permutation importance: {e}")
            return {}

    def _compute_and_aggregate_importance(self, subject_data):
        """Compute and aggregate permutation importance across all models and folds."""
        feature_scores = {}
        all_model_infos = []
        
        for rep in subject_data:
            for fold in rep:
                for model_info in fold:
                    all_model_infos.append(model_info)
        
        if not all_model_infos:
            return {}
        
        for model_info in all_model_infos:
            result = self._compute_permutation_importance_single_model(model_info)
            for feature_name, importance in result.items():
                if feature_name not in feature_scores:
                    feature_scores[feature_name] = []
                feature_scores[feature_name].append(importance)
        
        aggregated_importance = {}
        for name, scores in feature_scores.items():
            if scores:
                aggregated_importance[name] = np.mean(scores)
        
        return aggregated_importance

    def analyze_all_subjects(self):
        """Process all subjects and conditions, computing permutation importance."""
        subjects = self.get_available_subjects()
        logger.info(f"Found {len(subjects)} subjects: {subjects}")
        
        for condition in self.conditions:
            logger.info(f"Analyzing condition: {condition}")
            condition_importances = {}

            for subject_id in subjects:
                subject_data = self.load_subject_data(subject_id, condition)
                if subject_data is None:
                    continue
                
                agg_importance = self._compute_and_aggregate_importance(subject_data)
                if agg_importance:
                    condition_importances[subject_id] = agg_importance
            
            if condition_importances:
                results_df = pd.DataFrame.from_dict(condition_importances, orient='index')
                self.all_feature_importances[condition] = results_df.fillna(0)
                logger.info(f"Completed {condition}: {len(condition_importances)} subjects, {len(results_df.columns)} features")

    def save_importance_results(self, filename="feature_importance_results.csv"):
        """Save permutation importance results to CSV."""
        all_results = []
        
        for condition, df in self.all_feature_importances.items():
            mean_importances = df.mean(axis=0)
            for feature_name, importance in mean_importances.items():
                all_results.append({
                    "Condition": condition,
                    "Feature": feature_name,
                    "Mean_Permutation_Importance": importance
                })
        
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df = results_df.sort_values(by=["Condition", "Mean_Permutation_Importance"], ascending=[True, False])
            results_df.to_csv(filename, index=False)
            logger.info(f"Saved importance results to {filename}")
            return True
        return False

class EEGFeatureAnalyzer:
    """Analyzes feature distinctiveness between conditions."""
    
    def __init__(self):
        self.conditions = ['Single', 'SICI', 'SICF']
        self.brain_regions = {
            'l_front': ['F3', 'AF3', 'F1', 'F5', 'FC1', 'FC3'],
            'r_front': ['F4', 'AF4', 'F2', 'F6', 'FC2', 'FC4'],
            'm_front': ['Fz', 'FCz'],
            'l_cent': ['C3', 'C1', 'FC5', 'C5', 'CP1', 'CP3'],
            'r_cent': ['C4', 'C2', 'FC6', 'C6', 'CP2', 'CP4'],
            'm_cent': ['Cz', 'CPz'],
            'l_pari': ['P3', 'P1', 'P5', 'PO3', 'CP5'],
            'r_pari': ['P4', 'P2', 'P6', 'PO4', 'CP6'],
            'm_pari': ['Pz', 'POz'],
            'l_temp': ['T7', 'FT7', 'TP7'],
            'r_temp': ['T8', 'FT8', 'TP8'],
            'l_occip': ['O1', 'PO7'],
            'r_occip': ['O2', 'PO8'],
            'm_occip': ['Oz']
        }
        self.freq_bands = ['theta', 'alpha', 'beta', 'gamma']
        self.feature_types = ['pac', 'psd', 'wpli', 'phase']

    def categorize_feature(self, feature_name):
        """Categorize feature by brain region, frequency band, and feature type."""
        feature_name = feature_name.lower()
        
        brain_region = 'other'
        freq_band = 'other'
        feature_type = 'other'
        
        # Determine brain region
        for region, electrodes in self.brain_regions.items():
            for electrode in electrodes:
                if electrode.lower() in feature_name:
                    brain_region = region
                    break
            if brain_region != 'other':
                break
        
        # Determine frequency band
        for band in self.freq_bands:
            if band in feature_name:
                freq_band = band
                break
        
        # Determine feature type
        if 'pac' in feature_name:
            feature_type = 'pac'
        elif 'power_spectral_density' in feature_name or 'psd' in feature_name:
            feature_type = 'psd'
        elif 'wpli' in feature_name:
            feature_type = 'wpli'
        elif 'phase' in feature_name:
            feature_type = 'phase'
        
        return brain_region, freq_band, feature_type

    def create_feature_signature(self, brain_region, freq_band, feature_type):
        """Create a signature string for a feature combination."""
        return f"{brain_region}_{freq_band}_{feature_type}"

    def load_and_categorize_data(self, csv_path):
        """Load feature importance data and add categorizations."""
        self.importance_data = pd.read_csv(csv_path)
        
        brain_regions = []
        freq_bands = []
        feature_types = []
        feature_signatures = []
        
        for feature in self.importance_data['Feature']:
            brain_region, freq_band, feature_type = self.categorize_feature(feature)
            brain_regions.append(brain_region)
            freq_bands.append(freq_band)
            feature_types.append(feature_type)
            feature_signatures.append(self.create_feature_signature(brain_region, freq_band, feature_type))
        
        self.importance_data['brain_region'] = brain_regions
        self.importance_data['freq_band'] = freq_bands
        self.importance_data['feature_type'] = feature_types
        self.importance_data['feature_signature'] = feature_signatures
        
        logger.info(f"Loaded and categorized {len(self.importance_data)} features")

    def calculate_regional_preferences(self, min_importance=0.001):
        """Calculate brain region preferences for each condition."""
        filtered_data = self.importance_data[
            self.importance_data['Mean_Permutation_Importance'] > min_importance
        ]
        
        regional_stats = {}
        for condition in self.conditions:
            condition_data = filtered_data[filtered_data['Condition'] == condition]
            if len(condition_data) == 0:
                continue
            
            region_stats = condition_data.groupby('brain_region')['Mean_Permutation_Importance'].agg([
                'count', 'sum'
            ]).fillna(0)
            
            total_importance = condition_data['Mean_Permutation_Importance'].sum()
            region_stats['relative_importance'] = region_stats['sum'] / total_importance
            regional_stats[condition] = region_stats
        
        return regional_stats

    def calculate_frequency_preferences(self, min_importance=0.001):
        """Calculate frequency band preferences for each condition."""
        filtered_data = self.importance_data[
            self.importance_data['Mean_Permutation_Importance'] > min_importance
        ]
        
        frequency_stats = {}
        for condition in self.conditions:
            condition_data = filtered_data[filtered_data['Condition'] == condition]
            if len(condition_data) == 0:
                continue
            
            freq_stats = condition_data.groupby('freq_band')['Mean_Permutation_Importance'].agg([
                'count', 'sum'
            ]).fillna(0)
            
            total_importance = condition_data['Mean_Permutation_Importance'].sum()
            freq_stats['relative_importance'] = freq_stats['sum'] / total_importance
            frequency_stats[condition] = freq_stats
        
        return frequency_stats

    def calculate_signature_overlap(self, min_importance=0.001, top_n=10):
        """Calculate feature signature overlap between conditions based on region×freq×feature."""
        filtered_data = self.importance_data[
            (self.importance_data['Mean_Permutation_Importance'] > min_importance) &
            (self.importance_data['brain_region'] != 'other') &
            (self.importance_data['freq_band'] != 'other') &
            (self.importance_data['feature_type'] != 'other')
        ]
        
        # Get top signatures for each condition
        top_signatures_by_condition = {}
        for condition in self.conditions:
            condition_data = filtered_data[filtered_data['Condition'] == condition]
            if len(condition_data) == 0:
                continue
            
            # Aggregate importance by signature
            signature_importance = condition_data.groupby('feature_signature')['Mean_Permutation_Importance'].sum()
            top_signatures = signature_importance.nlargest(top_n).index.tolist()
            top_signatures_by_condition[condition] = set(top_signatures)
        
        # Calculate pairwise overlaps
        overlap_results = {}
        conditions = list(top_signatures_by_condition.keys())
        for i, cond1 in enumerate(conditions):
            for j, cond2 in enumerate(conditions):
                if i >= j:
                    continue
                
                sigs1 = top_signatures_by_condition[cond1]
                sigs2 = top_signatures_by_condition[cond2]
                
                intersection = sigs1.intersection(sigs2)
                union = sigs1.union(sigs2)
                
                jaccard = len(intersection) / len(union) if len(union) > 0 else 0
                overlap_pct = len(intersection) / min(len(sigs1), len(sigs2)) if min(len(sigs1), len(sigs2)) > 0 else 0
                
                overlap_results[f"{cond1}_{cond2}"] = {
                    'intersection_size': len(intersection),
                    'jaccard_similarity': jaccard,
                    'overlap_percentage': overlap_pct
                }
        
        return overlap_results, top_signatures_by_condition

    def perform_statistical_tests(self, regional_stats, frequency_stats):
        """Perform statistical tests for regional and frequency preferences."""
        results = {}
        
        # Test regional preferences
        if len(regional_stats) >= 2:
            all_regions = set()
            for stats in regional_stats.values():
                all_regions.update(stats.index)
            all_regions = [r for r in all_regions if r != 'other']
            
            if len(all_regions) > 1:
                contingency_data = []
                for condition in self.conditions:
                    if condition in regional_stats:
                        row = []
                        for region in all_regions:
                            count = regional_stats[condition].loc[region, 'count'] if region in regional_stats[condition].index else 0
                            row.append(int(count))
                        contingency_data.append(row)
                
                if len(contingency_data) > 1:
                    contingency_table = np.array(contingency_data)
                    try:
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                        n = contingency_table.sum()
                        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                        results['regional'] = {
                            'chi2': chi2,
                            'p_value': p_value,
                            'cramers_v': cramers_v,
                            'significant': p_value < 0.05
                        }
                    except ValueError:
                        results['regional'] = {'significant': False, 'cramers_v': 0.0}
        
        # Test frequency preferences
        if len(frequency_stats) >= 2:
            known_freqs = [f for f in self.freq_bands if f != 'other']
            
            if len(known_freqs) > 1:
                contingency_data = []
                for condition in self.conditions:
                    if condition in frequency_stats:
                        row = []
                        for freq in known_freqs:
                            count = frequency_stats[condition].loc[freq, 'count'] if freq in frequency_stats[condition].index else 0
                            row.append(int(count))
                        contingency_data.append(row)
                
                if len(contingency_data) > 1:
                    contingency_table = np.array(contingency_data)
                    try:
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                        n = contingency_table.sum()
                        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                        results['frequency'] = {
                            'chi2': chi2,
                            'p_value': p_value,
                            'cramers_v': cramers_v,
                            'significant': p_value < 0.05
                        }
                    except ValueError:
                        results['frequency'] = {'significant': False, 'cramers_v': 0.0}
        
        return results

def create_regional_preference_plot(ax, regional_stats):
    """Create brain region preference plot with laterality."""
    if not regional_stats:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        return
    
    all_regions = set()
    for stats in regional_stats.values():
        all_regions.update(stats.index)
    all_regions = [r for r in sorted(all_regions) if r != 'other']
    
    if not all_regions:
        ax.text(0.5, 0.5, 'No regional data', ha='center', va='center')
        return
    
    conditions = list(regional_stats.keys())
    x = np.arange(len(all_regions))
    width = 0.25
    colors = ['#cee1d1', '#80a687', '#9e9e9e']
    
    for i, condition in enumerate(conditions):
        if condition in regional_stats:
            values = []
            for region in all_regions:
                if region in regional_stats[condition].index:
                    values.append(regional_stats[condition].loc[region, 'relative_importance'])
                else:
                    values.append(0)
            
            ax.bar(x + i * width, values, width, label=condition, 
                  color=colors[i % len(colors)], 
                  edgecolor=darken_color(colors[i % len(colors)]), linewidth=0.5)
    
    ax.set_xlabel('Brain region')
    ax.set_ylabel('Relative importance')
    ax.set_title('Regional preferences')
    ax.set_xticks(x + width)


    
    formatted_labels = []
    for region in all_regions:
        parts = region.split('_')
        if len(parts) >= 2:
            formatted_labels.append(f"{parts[0][0].upper()}-{parts[1][0].upper()}")
        else:
            formatted_labels.append(region.title())
    
    # rotate xticks 45 degrees
    ax.set_xticklabels(formatted_labels, rotation=45, ha='center', fontsize=6)
    ax.legend(loc='best', fontsize=6)
    ax.set_ylim(0, max([max([regional_stats[c].loc[r, 'relative_importance'] 
                            for r in all_regions if r in regional_stats[c].index] + [0])
                       for c in conditions if c in regional_stats] + [0.1]) * 1.1)

def create_frequency_preference_plot(ax, frequency_stats, conditions_order, condition_colors):
    """Create frequency band preference plot."""
    if not frequency_stats:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        return
    
    freq_bands = ['theta', 'alpha', 'beta', 'gamma']
    x = np.arange(len(freq_bands))
    width = 0.25
    
    for i, condition in enumerate(conditions_order):
        if condition in frequency_stats:
            values = []
            for freq in freq_bands:
                if freq in frequency_stats[condition].index:
                    values.append(frequency_stats[condition].loc[freq, 'relative_importance'])
                else:
                    values.append(0)
            
            color = condition_colors.get(condition, '#cccccc')
            ax.bar(x + i * width, values, width, label=condition, 
                  color=color, 
                  edgecolor=darken_color(color), linewidth=0.5)
    
    ax.set_xlabel('Frequency band')
    ax.set_ylabel('Relative importance')
    ax.set_title('Frequency preferences')
    ax.set_xticks(x + width)
    ax.set_xticklabels([b.capitalize() for b in freq_bands])
    
    max_y = 0
    for cond in conditions_order:
        if cond in frequency_stats:
            max_y = max(max_y, frequency_stats[cond]['relative_importance'].max())
    ax.set_ylim(0, max_y * 1.15 if max_y > 0 else 0.1)

def create_signature_overlap_heatmap(ax, overlap_results, conditions):
    """Create feature signature overlap heatmap."""
    if not overlap_results:
        ax.text(0.5, 0.5, 'No overlap data', ha='center', va='center')
        return
    
    n_conditions = len(conditions)
    overlap_matrix = np.eye(n_conditions)
    
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            if i != j:
                key = f"{cond1}_{cond2}" if f"{cond1}_{cond2}" in overlap_results else f"{cond2}_{cond1}"
                if key in overlap_results:
                    overlap_matrix[i, j] = overlap_results[key]['jaccard_similarity']
    
    im = ax.imshow(overlap_matrix, cmap='Greens', vmin=0, vmax=1)
    
    ax.set_xticks(range(n_conditions))
    ax.set_yticks(range(n_conditions))
    ax.set_xticklabels(conditions)
    ax.set_yticklabels(conditions)
    ax.set_title('Feature signature overlap\n(region×freq×feature)')
    
    for i in range(n_conditions):
        for j in range(n_conditions):
            text = ax.text(j, i, f'{overlap_matrix[i, j]:.2f}',
                         ha="center", va="center", 
                         color="white" if overlap_matrix[i, j] > 0.5 else "black",
                         fontsize=6)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Jaccard Similarity', rotation=270, labelpad=15)

def create_top_features_plot(ax, importance_data, condition, condition_colors, n_top=10):
    """Create a horizontal bar plot of top features for a given condition."""
    condition_data = importance_data[importance_data['Condition'] == condition]
    
    if condition_data.empty:
        ax.text(0.5, 0.5, f'No data for {condition}', ha='center', va='center')
        return

    top_features = condition_data.nlargest(n_top, 'Mean_Permutation_Importance')
    
    def format_feature_label(label):
        """Format complex feature names into a readable, compact string."""
        freq_map = {'theta': 'θ', 'alpha': 'α', 'beta': 'β', 'gamma': 'γ'}
        
        label = label.replace('power_spectral_density', 'psd')
        if 'phase' in label:
            label = label.replace('_sin', '').replace('_cos', '')

        parts = label.split('_')
        
        if len(parts) > 2:
            feature_type = parts[0]
            freq_band_raw = parts[1]
            freq_symbol = freq_map.get(freq_band_raw, freq_band_raw)
            
            electrode_info = '_'.join(parts[2:])

            if 'hjorth' in electrode_info and len(parts) > 4:
                e1 = parts[2]
                e2 = parts[4]
                electrode_formatted = f"{e1}-{e2}_hjorth"
            else:
                electrode_formatted = electrode_info.replace('_', '-')
            
            return f"{freq_symbol}-{feature_type} {electrode_formatted}"
            
        return label

    y_pos = np.arange(len(top_features))
    feature_labels = [format_feature_label(f) for f in top_features['Feature']]
    color = condition_colors.get(condition, '#80a687')

    ax.barh(y_pos, top_features['Mean_Permutation_Importance'], 
            color=color, edgecolor=darken_color(color), linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_labels, fontsize=6)
    ax.invert_yaxis()

    max_val = top_features['Mean_Permutation_Importance'].max()
    ax.set_xlim(0, max_val * 1.05)
    ax.set_xticks([0, max_val])
    ax.set_xticklabels(['0', f'{max_val:.4f}'])
    
    ax.set_xlabel('Permutation importance')
    ax.set_title(f'Top {n_top} features - {condition}')
    ax.tick_params(axis='y', which='major', pad=2)

    pos = ax.get_position()
    ax.set_position([pos.x0 + 0.1, pos.y0, pos.width - 0.1, pos.height])

def main():
    """Main function to run the complete analysis."""
    RESULTS_DIR = "/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/SICISICF/results/models"
    CSV_FILE = "feature_importance_results.csv"
    
    apply_matplotlib_settings()
    
    if not Path(CSV_FILE).exists():
        logger.info("CSV file not found. Computing permutation importance...")
        
        if not Path(RESULTS_DIR).exists():
            logger.error(f"Results directory not found: {RESULTS_DIR}")
            logger.error("Please update RESULTS_DIR or provide the CSV file directly.")
            return
        
        analyzer = EEGPermutationAnalyzer(RESULTS_DIR)
        analyzer.analyze_all_subjects()
        
        if not analyzer.save_importance_results(CSV_FILE):
            logger.error("Failed to save permutation importance results")
            return
        
        logger.info(f"Permutation importance computed and saved to {CSV_FILE}")
    
    feature_analyzer = EEGFeatureAnalyzer()
    feature_analyzer.load_and_categorize_data(CSV_FILE)

    conditions_order = ['Single', 'SICI', 'SICF']
    condition_colors = {
        'Single': '#cee1d1',
        'SICI': '#80a687',
        'SICF': '#9e9e9e'
    }
    
    regional_stats = feature_analyzer.calculate_regional_preferences()
    frequency_stats = feature_analyzer.calculate_frequency_preferences()
    overlap_results, top_signatures = feature_analyzer.calculate_signature_overlap()
    stat_results = feature_analyzer.perform_statistical_tests(regional_stats, frequency_stats)
    
    fig = plt.figure(figsize=(195 / 25.4, 120 / 25.4))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)
    
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[0, 2])
    ax_D = fig.add_subplot(gs[1, 0])
    ax_E = fig.add_subplot(gs[1, 1])
    ax_F = fig.add_subplot(gs[1, 2])
    
    create_regional_preference_plot(ax_A, regional_stats)
    create_frequency_preference_plot(ax_B, frequency_stats, conditions_order, condition_colors)
    create_signature_overlap_heatmap(ax_C, overlap_results, feature_analyzer.conditions)
    create_top_features_plot(ax_D, feature_analyzer.importance_data, 'Single', condition_colors, n_top=10)
    create_top_features_plot(ax_E, feature_analyzer.importance_data, 'SICI', condition_colors, n_top=10)
    create_top_features_plot(ax_F, feature_analyzer.importance_data, 'SICF', condition_colors, n_top=10)
    
    for ax, label in zip([ax_A, ax_B, ax_C, ax_D, ax_E, ax_F], ['a', 'b', 'c', 'd', 'e', 'f']):
        ax.text(-0.2, 1.1, label, transform=ax.transAxes, fontsize=9, 
                fontweight='bold', va='top', ha='right')
    
    if 'regional' in stat_results and stat_results['regional']['significant']:
        stars = get_significance_stars(stat_results['regional']['p_value'])
        ax_A.text(0.95, 0.95, stars, transform=ax_A.transAxes, 
                 fontsize=8, fontweight='bold', va='top', ha='right')
    
    if 'frequency' in stat_results and stat_results['frequency']['significant']:
        stars = get_significance_stars(stat_results['frequency']['p_value'])
        ax_B.text(0.95, 0.95, stars, transform=ax_B.transAxes, 
                 fontsize=8, fontweight='bold', va='top', ha='right')
    
    plt.tight_layout()
    
    output_filename = "eeg_feature_distinctiveness_analysis.pdf"
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.2, dpi=300)
    plt.savefig(output_filename.replace('.pdf', '.png'), dpi=300)
    logger.info(f"Figure saved as {output_filename}")
    
    print("\n" + "="*60)
    print("EEG FEATURE DISTINCTIVENESS ANALYSIS - SUMMARY")
    print("="*60)
    
    if 'regional' in stat_results:
        print(f"Regional distinctiveness (Cramer's V): {stat_results['regional']['cramers_v']:.3f}")
        if stat_results['regional']['significant']:
            print(f"Regional differences: SIGNIFICANT (p = {stat_results['regional']['p_value']:.4f})")
        else:
            print("Regional differences: Not significant")
    
    if 'frequency' in stat_results:
        print(f"Frequency distinctiveness (Cramer's V): {stat_results['frequency']['cramers_v']:.3f}")
        if stat_results['frequency']['significant']:
            print(f"Frequency differences: SIGNIFICANT (p = {stat_results['frequency']['p_value']:.4f})")
        else:
            print("Frequency differences: Not significant")
    
    if overlap_results:
        overlaps = [result['jaccard_similarity'] for result in overlap_results.values()]
        mean_overlap = np.mean(overlaps)
        print(f"Mean signature overlap (Jaccard): {mean_overlap:.3f}")
        
        for comparison, stats in overlap_results.items():
            conditions = comparison.split('_')
            print(f"  {conditions[0]} vs {conditions[1]}: {stats['jaccard_similarity']:.3f}")
    
    if overlap_results and 'regional' in stat_results and 'frequency' in stat_results:
        mean_overlap = np.mean([result['jaccard_similarity'] for result in overlap_results.values()])
        regional_dist = stat_results['regional']['cramers_v']
        freq_dist = stat_results['frequency']['cramers_v']
        
        distinctiveness_score = (1 - mean_overlap) * 0.4 + regional_dist * 0.3 + freq_dist * 0.3
        
        print(f"\nOverall distinctiveness score: {distinctiveness_score:.3f}")
        
        if distinctiveness_score > 0.6:
            conclusion = "CONDITIONS RELY ON DISTINCT PREDICTIVE SIGNATURES"
        elif distinctiveness_score > 0.4:
            conclusion = "CONDITIONS SHOW PARTIAL DISTINCTIVENESS"
        else:
            conclusion = "CONDITIONS RELY ON OVERLAPPING PREDICTIVE SIGNATURES"
        
        print(f"CONCLUSION: {conclusion}")
    
    print("="*60)
    
    plt.show()

if __name__ == "__main__":
    main()
# %%