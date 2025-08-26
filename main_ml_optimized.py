#%%
"""
Main execution script for EEG classification analysis.
Implements Bayesian optimization for hyperparameter tuning and LOSO transfer learning.
"""

import os
import sys
from pathlib import Path
import numpy as np
from datetime import datetime
import time

# Add source directory to path
sys.path.append('src')

from src.Dependencies import *
from src.mRMR_feature_select import mRMR_feature_select
from src.helper_functions import *
from src.config_management import get_config


def validate_packages():
    """Ensure required packages are available."""
    try:
        import optuna
        from scipy.stats import uniform, randint, loguniform
        return True
    except ImportError as e:
        print(f"Required package missing: {e}")
        return False


def analyze_dataset(features_path):
    """Analyze dataset characteristics and provide summary."""
    print("Dataset Analysis")
    print("-" * 40)
    
    subjects = get_available_subjects(features_path)
    if not subjects:
        print("No subjects found")
        return None
    
    # Analyze sample subject
    sample_subject = subjects[0]
    conditions = ['SICI', 'SICF', 'Single']
    
    for condition in conditions:
        try:
            df = load_subject_data(sample_subject, features_path, condition)
            if df is not None:
                feature_cols = [col for col in df.columns 
                              if col not in ['Trial_index', 'Condition_name', 'Label', 'Condition']]
                
                print(f"{condition} condition:")
                print(f"  Total trials: {len(df)}")
                print(f"  Feature count: {len(feature_cols)}")
                print(f"  Class distribution: {df['Condition'].value_counts().to_dict()}")
                
                min_class = df['Condition'].value_counts().min()
                print(f"  Minimum class size: {min_class}")
                
                if min_class < 15:
                    print(f"  Warning: Small class size may lead to unstable results")
                if len(df) < 100:
                    print(f"  Warning: Small dataset - consider reducing optimization budget")
                
        except Exception as e:
            print(f"  Could not analyze {condition}: {e}")
    
    return subjects


def run_single_subject_analysis(subject_id, features_path, condition_filter, config, analysis_params):
    """Run analysis for a single subject and condition."""
    print(f"Processing subject {subject_id}, condition: {condition_filter}")
    
    # Load data
    data = load_subject_data(subject_id, features_path, condition_filter)
    if data is None:
        print(f"Could not load data for {subject_id}")
        return False
        
    # Check sample sizes
    condition_counts = data['Condition'].value_counts()
    print(f"Class distribution: {condition_counts.to_dict()}")
    
    if len(condition_counts) < 2 or min(condition_counts) < 10:
        print(f"Insufficient samples for {subject_id}, condition {condition_filter}")
        print(f"Need at least 10 samples per class, got: {condition_counts.to_dict()}")
        return False
    
    # Validate dataset
    feature_cols = [col for col in data.columns 
                   if col not in ["Trial_index", "Condition_name", "Label", "Condition"]]
    
    n_samples = len(data)
    n_trials = config.optimization.n_trials
    
    print(f"Dataset validation:")
    print(f"  Total samples: {n_samples}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Optimization trials per model: {n_trials}")
    
    if n_samples < 50:
        print(f"  Warning: Very small dataset ({n_samples} samples)")
    elif n_samples < 100:
        print(f"  Warning: Small dataset ({n_samples} samples)")
    else:
        print(f"  Dataset size suitable for optimization")
    
    # Add condition filter to analysis params
    analysis_params['condition_filter'] = condition_filter
    
    # Run analysis
    try:
        run_cross_validation_analysis(data, subject_id, config, analysis_params)
        print(f"Successfully completed {subject_id} - {condition_filter}")
        return True
    except Exception as e:
        print(f"Error in {subject_id} - {condition_filter}: {e}")
        return False


def run_condition_analysis(subjects, features_path, condition, variation_params, config):
    """Run analysis for all subjects in a specific condition."""
    print(f"Running analysis: {condition} - {variation_params}")
    
    success_count = 0
    total_count = len(subjects)
    
    for subject_id in subjects:
        success = run_single_subject_analysis(
            subject_id, features_path, condition, config, variation_params
        )
        if success:
            success_count += 1
    
    print(f"Completed {condition}: {success_count}/{total_count} subjects successful")
    return success_count, total_count


def run_full_analysis(subjects, features_path, config):
    """Run complete analysis for all conditions and variations."""
    print("Starting full analysis")
    print("-" * 40)
    
    conditions = ['SICI', 'SICF', 'Single']
    variations = {
        'standard': {'shuffle': False, 'time_based': False},
        'shuffled': {'shuffle': True, 'time_based': False},
        'temporal': {'shuffle': False, 'time_based': True}
    }
    
    if not subjects:
        print("No subjects available for analysis")
        return
    
    total_analyses = len(conditions) * len(variations)
    current_analysis = 0
    results_summary = []
    
    start_time = datetime.now()
    
    for condition in conditions:
        for variation_name, variation_params in variations.items():
            current_analysis += 1
            analysis_name = f"{condition}-{variation_name}"
            
            print(f"Analysis {current_analysis}/{total_analyses}: {analysis_name}")
            
            # Estimate remaining time
            if current_analysis > 1:
                elapsed = datetime.now() - start_time
                avg_time = elapsed / (current_analysis - 1)
                remaining = avg_time * (total_analyses - current_analysis)
                print(f"Estimated remaining time: {remaining}")
            
            success_count, total_count = run_condition_analysis(
                subjects, features_path, condition, variation_params, config
            )
            
            results_summary.append({
                'analysis': analysis_name,
                'success': success_count,
                'total': total_count,
                'success_rate': success_count / total_count if total_count > 0 else 0
            })
    
    # Print summary
    total_time = datetime.now() - start_time
    print(f"Full analysis completed in {total_time}")
    print("Results summary:")
    
    for result in results_summary:
        success_rate = result['success_rate'] * 100
        print(f"  {result['analysis']}: {result['success']}/{result['total']} ({success_rate:.1f}%)")
    
    print(f"Results saved to: {config.paths.model_savepath}")


def run_loso_analysis(features_path, config, condition, variation_params):
    """Run LOSO transfer learning analysis."""
    print(f"Running LOSO analysis: {condition} - {variation_params}")
    
    try:
        results = run_loso_transfer_learning(
            features_path=features_path,
            config=config,
            condition_filter=condition,
            analysis_params=variation_params
        )
        
        if results:
            print_loso_summary(results, condition)
            return results
        else:
            print("LOSO analysis failed - no results returned")
            return None
            
    except Exception as e:
        print(f"Error in LOSO analysis: {e}")
        return None


def print_loso_summary(results, condition):
    """Print summary of LOSO transfer learning results."""
    print(f"LOSO Summary: {condition}")
    print("-" * 40)
    
    zero_shot_results = results['zero_shot_results']
    calibrated_results = results['calibrated_results']
    bayesian_info = results['bayesian_optimization_info']
    feature_info = results['feature_selection_info']
    
    model_names = ['SVM', 'LogReg', 'RF']
    
    # Calculate averages
    for model_name in model_names:
        zero_shot_accs = []
        calibrated_accs = []
        source_cv_scores = []
        
        for subj in zero_shot_results.keys():
            if model_name in zero_shot_results[subj]:
                zero_shot_accs.append(zero_shot_results[subj][model_name]['accuracy'])
                calibrated_accs.append(calibrated_results[subj][model_name]['accuracy'])
                source_cv_scores.append(zero_shot_results[subj][model_name]['source_cv_score'])
        
        if zero_shot_accs and calibrated_accs:
            zero_shot_mean = np.mean(zero_shot_accs)
            zero_shot_std = np.std(zero_shot_accs)
            calibrated_mean = np.mean(calibrated_accs)
            calibrated_std = np.std(calibrated_accs)
            source_cv_mean = np.mean(source_cv_scores)
            improvement = calibrated_mean - zero_shot_mean
            
            print(f"{model_name}:")
            print(f"  Source CV:  {source_cv_mean:.3f}")
            print(f"  0-shot:     {zero_shot_mean:.3f} ± {zero_shot_std:.3f}")
            print(f"  Calibrated: {calibrated_mean:.3f} ± {calibrated_std:.3f}")
            print(f"  Improvement: {improvement:+.3f}")
    
    # Feature selection summary
    feature_counts = [info['selected_features_count'] for info in feature_info.values()]
    if feature_counts:
        avg_features = np.mean(feature_counts)
        print(f"Average features selected: {avg_features:.1f}")


def run_full_loso_analysis(subjects, features_path, config):
    """Run complete LOSO analysis for all conditions and variations."""
    print("Starting full LOSO analysis")
    print("-" * 40)
    
    if len(subjects) < 3:
        print(f"Need at least 3 subjects for LOSO, got {len(subjects)}")
        return
    
    conditions = ['SICI', 'SICF', 'Single']
    variations = {
        'standard': {'shuffle': False, 'time_based': False},
        'shuffled': {'shuffle': True, 'time_based': False},
        'temporal': {'shuffle': False, 'time_based': True}
    }
    
    total_analyses = len(conditions) * len(variations)
    current_analysis = 0
    completed_analyses = []
    failed_analyses = []
    
    start_time = datetime.now()
    
    for condition in conditions:
        for variation_name, variation_params in variations.items():
            current_analysis += 1
            analysis_name = f"LOSO-{condition}-{variation_name}"
            
            print(f"LOSO Analysis {current_analysis}/{total_analyses}: {analysis_name}")
            
            # Estimate remaining time
            if current_analysis > 1:
                elapsed = datetime.now() - start_time
                avg_time = elapsed / (current_analysis - 1)
                remaining = avg_time * (total_analyses - current_analysis)
                print(f"Estimated remaining time: {remaining}")
            
            results = run_loso_analysis(features_path, config, condition, variation_params)
            
            if results:
                completed_analyses.append(analysis_name)
                print(f"Completed {analysis_name}")
            else:
                failed_analyses.append(analysis_name)
                print(f"Failed {analysis_name}")
    
    # Summary
    total_time = datetime.now() - start_time
    print(f"Full LOSO analysis completed in {total_time}")
    print(f"Completed: {len(completed_analyses)}/{total_analyses}")
    print(f"Failed: {len(failed_analyses)}/{total_analyses}")
    
    if completed_analyses:
        print("Successful analyses:")
        for analysis in completed_analyses:
            print(f"  {analysis}")
    
    if failed_analyses:
        print("Failed analyses:")
        for analysis in failed_analyses:
            print(f"  {analysis}")
    
    print(f"Results saved to: {config.paths.model_savepath}")


def test_single_subject():
    """Test analysis with a single subject and condition."""
    if not subjects:
        print("No subjects available for testing")
        return
    
    test_subject = subjects[0]
    test_condition = 'SICI'
    
    print(f"Testing with subject: {test_subject}, condition: {test_condition}")
    
    success = run_single_subject_analysis(
        test_subject, config.paths.features_path, test_condition, config,
        {'shuffle': False, 'time_based': False}
    )
    
    if success:
        print("Single subject test completed successfully")
    else:
        print("Single subject test failed")


def test_loso_pipeline():
    """Test LOSO transfer learning pipeline."""
    print("Testing LOSO transfer learning pipeline")
    
    if len(subjects) < 3:
        print("Need at least 3 subjects for LOSO transfer learning")
        return
    
    results = run_loso_analysis(
        config.paths.features_path, config, 'SICI', 
        {'shuffle': False, 'time_based': False}
    )
    
    if results:
        print("LOSO test completed successfully")
        
        # Verify results structure
        required_keys = ['zero_shot_results', 'calibrated_results', 'bayesian_optimization_info']
        for key in required_keys:
            if key in results:
                print(f"  Found {key}")
            else:
                print(f"  Missing {key}")
    else:
        print("LOSO test failed")


def compare_approaches():
    """Compare within-subject CV vs LOSO transfer learning."""
    print("Comparing within-subject CV vs LOSO transfer learning")
    print("-" * 50)
    
    condition = 'SICI'
    test_subjects = subjects[:3]
    
    # Run standard analysis
    print("1. Running within-subject analysis...")
    run_condition_analysis(
        test_subjects, config.paths.features_path, condition,
        {'shuffle': False, 'time_based': False}, config
    )
    
    # Run LOSO analysis
    print("2. Running LOSO transfer learning...")
    loso_results = run_loso_analysis(
        config.paths.features_path, config, condition,
        {'shuffle': False, 'time_based': False}
    )
    
    print("Comparison completed")
    print("Key differences:")
    print("  Within-subject: Individual optimization per subject")
    print("  LOSO: Transfer learning with consistent feature sets")


# Initialize configuration and validate environment
print("EEG Classification Analysis Pipeline")
print("=" * 50)

if not validate_packages():
    print("Required packages missing. Please install optuna and scipy.")
    sys.exit(1)

config = get_config()
print(f"Configuration loaded:")
print(f"  Optimization method: {config.optimization.method}")
print(f"  Optimization trials: {config.optimization.n_trials} per model")
print(f"  Cross-validation: {config.cross_validation.n_repetitions} reps × {config.cross_validation.k_out} folds")
print(f"  Max mRMR features: {config.feature_selection.max_mRMR_features}")

# Analyze dataset
subjects = analyze_dataset(config.paths.features_path)

if not subjects:
    print("No subjects found. Please check your features path.")
    print(f"Looking in: {config.paths.features_path}")
    sys.exit(1)

print(f"Found {len(subjects)} subjects: {subjects}")

# Estimate computational requirements
total_estimated_trials = (config.optimization.n_trials * 3 *
                         config.cross_validation.n_repetitions * 
                         config.cross_validation.k_out)
print(f"Estimated optimization trials per condition: {total_estimated_trials}")

# Available execution functions
available_functions = {
    'test_single_subject': test_single_subject,
    'test_loso_pipeline': test_loso_pipeline,
    'run_condition_analysis': lambda: run_condition_analysis(
        subjects, config.paths.features_path, 'SICI', 
        {'shuffle': False, 'time_based': False}, config
    ),
    'run_full_analysis': lambda: run_full_analysis(subjects, config.paths.features_path, config),
    'run_loso_analysis': lambda: run_loso_analysis(
        config.paths.features_path, config, 'SICI',
        {'shuffle': False, 'time_based': False}
    ),
    'run_full_loso_analysis': lambda: run_full_loso_analysis(subjects, config.paths.features_path, config),
    'compare_approaches': compare_approaches
}

print("\nAvailable execution options:")
for i, func_name in enumerate(available_functions.keys(), 1):
    print(f"{i}. {func_name}()")

# Main execution
if __name__ == "__main__":
    # Default execution - run single subject test
    print("\nRunning single subject test...")
    #test_single_subject()
    run_condition_analysis(subjects, config.paths.features_path, 'SICF', {'shuffle': False, 'time_based': False}, config)
    run_condition_analysis(subjects, config.paths.features_path, 'SICI', {'shuffle': False, 'time_based': False}, config)
    run_condition_analysis(subjects, config.paths.features_path, 'Single', {'shuffle': False, 'time_based': False}, config)
    run_loso_analysis(config.paths.features_path, config, 'SICI', {'shuffle': False, 'time_based': False})
    run_loso_analysis(config.paths.features_path, config, 'SICF', {'shuffle': False, 'time_based': False})
    run_loso_analysis(config.paths.features_path, config, 'Single', {'shuffle': False, 'time_based': False})
# %%

