"""
Helper functions for EEG classification analysis.
Includes data loading, feature selection, model optimization, and evaluation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os
import json
import pickle
import time
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from collections import Counter
import optuna
from optuna.samplers import TPESampler
from scipy.stats import uniform, randint

from Dependencies import *
from mRMR_feature_select import mRMR_feature_select
from config_management import *

GLOBAL_SEED = 42


def load_features_new_format(features_path: str, subject_id: str, condition_filter: str = None) -> pd.DataFrame:
    """Load and concatenate all feature files for a subject from new format."""
    features_path = Path(features_path)
    
    # Load binary labels and trial info
    labels_file = features_path / f"{subject_id}_binary_labels.npz"
    trial_info_file = features_path / f"{subject_id}_trial_info.npz"
    
    if not labels_file.exists() or not trial_info_file.exists():
        raise FileNotFoundError(f"Required files not found for {subject_id}")
    
    labels_data = np.load(labels_file, allow_pickle=True)
    trial_data = np.load(trial_info_file, allow_pickle=True)
    
    conditions = trial_data['conditions']
    trial_indices = trial_data['trial_indices']
    labels = labels_data['labels']
    
    # Create masks for condition filtering and valid labels
    if condition_filter:
        condition_mask = conditions == condition_filter
    else:
        condition_mask = np.ones(len(conditions), dtype=bool)
    
    valid_label_mask = ~np.isnan(labels)
    combined_mask = condition_mask & valid_label_mask
    
    # Create base dataframe
    base_df = pd.DataFrame({
        'Trial_index': trial_indices[combined_mask],
        'Condition_name': conditions[combined_mask],
        'Label': labels[combined_mask].astype(int)
    })
    
    # Load feature files
    feature_files = {
        'psd': f"{subject_id}_psd_features.npz",
        'wpli': f"{subject_id}_hjorth_wpli_features.npz", 
        'pac': f"{subject_id}_hjorth_pac_features.npz",
        'phase': f"{subject_id}_hjorth_phase_features.npz"
    }

    all_features = []
    
    for feature_type, filename in feature_files.items():
        filepath = features_path / filename
        if filepath.exists():
            feature_data = np.load(filepath, allow_pickle=True)
            
            for feature_name, feature_array in feature_data.items():
                # Squeeze leading dimension if present
                if feature_array.ndim > 1 and feature_array.shape[0] == 1:
                    feature_array = feature_array.squeeze(0)
                
                # Apply combined mask
                feature_array_filtered = feature_array[combined_mask]
                
                # Handle different array dimensions
                if feature_array_filtered.ndim == 1:
                    all_features.append(pd.DataFrame({feature_name: feature_array_filtered}))
                elif feature_array_filtered.ndim == 2:
                    col_names = [f"{feature_name}_{i}" for i in range(feature_array_filtered.shape[1])]
                    all_features.append(pd.DataFrame(feature_array_filtered, columns=col_names))
                else:
                    # Flatten higher-dimensional features
                    n_epochs = feature_array_filtered.shape[0]
                    reshaped_features = feature_array_filtered.reshape(n_epochs, -1)
                    col_names = [f"{feature_name}_{i}" for i in range(reshaped_features.shape[1])]
                    all_features.append(pd.DataFrame(reshaped_features, columns=col_names))
    
    # Concatenate all features
    if all_features:
        features_df = pd.concat([base_df.reset_index(drop=True)] + 
                               [f_df.reset_index(drop=True) for f_df in all_features], axis=1)
    else:
        features_df = base_df
    
    # Clean data
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_df['Condition'] = features_df['Label'].astype(int)
    
    return features_df


def load_subject_data(subject_id, features_path, condition_filter=None):
    """Load subject data from new feature format."""
    try:
        return load_features_new_format(features_path, subject_id, condition_filter)
    except (FileNotFoundError, Exception) as e:
        print(f"Subject {subject_id} condition {condition_filter} not found: {e}")
        return None


def prepare_train_test_data(train_index, test_index, data, feature_cols, target_col):
    """Split and standardize data for training and testing."""
    X_train = data.iloc[train_index][feature_cols]
    X_test = data.iloc[test_index][feature_cols]
    y_train = data.iloc[train_index][target_col]
    y_test = data.iloc[test_index][target_col]

    # Robust scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_test = pd.DataFrame(X_test_scaled, columns=feature_cols)

    return X_train, X_test, y_train, y_test, scaler


def check_data_alignment(X_train, y_train, X_test, y_test):
    """Ensure alignment between features and labels."""
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Mismatch in number of rows between features and labels.")

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, y_train, X_test, y_test


def select_features_by_type(X_train, y_train, feature_types, max_features):
    """Perform mRMR feature selection for each feature type."""
    selected_features = []

    for feature_type in feature_types:
        type_features = [col for col in X_train.columns if feature_type in col]
        
        if not type_features:
            continue
            
        X_type = X_train[type_features]

        if X_type.shape[1] <= max_features:
            selected_features.extend(type_features)
        else:
            mrmr_indices = mRMR_feature_select(
                X_type.values, y_train.values,
                num_features_to_select=max_features,
                K_MAX=500, n_jobs=-1, verbose=False
            )
            selected_features.extend(X_type.columns[mrmr_indices].tolist())

    return list(set(selected_features))


def select_features_global(X_train, y_train, max_features):
    """Global mRMR feature selection across all features."""
    all_features = X_train.columns.tolist()
    num_to_select = min(len(all_features), max_features)

    if len(all_features) <= num_to_select:
        return all_features

    mrmr_indices = mRMR_feature_select(
        X_train.values, y_train.values,
        num_features_to_select=num_to_select,
        K_MAX=300, n_jobs=-1, verbose=False
    )
    
    return X_train.columns[mrmr_indices].tolist()


def get_cv_folds(config):
    """Get cross-validation folds from config."""
    return config.cross_validation.k_in


def create_inner_cv_splits(X_train, y_train, k_fold, rep, random_state):
    """Generate inner CV folds with error handling."""
    try:
        skf = StratifiedKFold(n_splits=k_fold, shuffle=True, 
                            random_state=random_state + rep + 1)
        splits = [(train_index, test_index) 
                 for train_index, test_index in skf.split(X_train, y_train)]
        
        # Validate splits
        for i, (train_idx, val_idx) in enumerate(splits):
            if len(train_idx) == 0 or len(val_idx) == 0:
                raise ValueError(f"Empty split detected in fold {i}")
            
            y_val_fold = y_train.iloc[val_idx]
            if len(y_val_fold.unique()) < 2:
                print(f"Warning: Fold {i} validation set has only one class")
        
        return splits
        
    except Exception as e:
        print(f"Error creating inner folds: {e}")
        # Fallback split
        n_samples = len(X_train)
        split_point = int(0.8 * n_samples)
        train_idx = np.arange(split_point)
        val_idx = np.arange(split_point, n_samples)
        return [(train_idx, val_idx)]


def get_available_subjects(features_path):
    """Detect available subjects from features directory."""
    features_path = Path(features_path)
    subjects = []
    
    for labels_file in features_path.glob("*_binary_labels.npz"):
        subject_id = labels_file.name.replace("_binary_labels.npz", "")
        subjects.append(subject_id)
    
    return sorted(subjects)


def precompute_fold_features(X_train, y_train, cv_splits, max_features):
    """Pre-compute mRMR feature selection for all CV folds."""
    fold_features = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        
        # Clean data for mRMR
        X_fold_clean = X_fold_train.fillna(X_fold_train.median())
        X_fold_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_fold_clean = X_fold_clean.fillna(X_fold_clean.median())
        
        # Select features for this fold
        n_features = min(max_features, X_fold_clean.shape[1])
        
        if n_features >= X_fold_clean.shape[1]:
            selected = X_fold_clean.columns.tolist()
        else:
            mrmr_indices = mRMR_feature_select(
                X_fold_clean.values, y_fold_train.values,
                num_features_to_select=n_features,
                K_MAX=300, n_jobs=-1, verbose=False
            )
            selected = X_fold_clean.columns[mrmr_indices].tolist()
        
        fold_features.append(selected)
    
    return fold_features


def optimize_svm_bayesian(X_train, y_train, cv_splits, fold_features, config):
    """Bayesian optimization for SVM with pre-computed features."""
    bayesian_config = config.bayesian_ranges.SVM
    n_trials = config.optimization.n_trials
    startup_trials = config.optimization.startup_trials

    def objective(trial):
        C = trial.suggest_float('C', bayesian_config.C_range[0], 
                               bayesian_config.C_range[1], log=True)
        kernel = trial.suggest_categorical('kernel', bayesian_config.kernels)
        
        gamma = "scale"
        if kernel == 'rbf':
            gamma = trial.suggest_float("gamma", 1e-4, 1e1, log=True)
        
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            selected_features = fold_features[fold_idx]
            
            X_fold_train = X_train.iloc[train_idx][selected_features]
            X_fold_val = X_train.iloc[val_idx][selected_features]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            scaler = RobustScaler()
            X_fold_train_scaled = scaler.fit_transform(X_fold_train)
            X_fold_val_scaled = scaler.transform(X_fold_val)
            
            svm = SVC(C=C, kernel=kernel, gamma=gamma, random_state=GLOBAL_SEED, tol=1e-3)
            svm.fit(X_fold_train_scaled, y_fold_train)
            
            score = svm.score(X_fold_val_scaled, y_fold_val)
            fold_scores.append(score)
            
            trial.report(score, fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return float(np.mean(fold_scores))

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1)
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42, n_startup_trials=startup_trials, multivariate=True),
        pruner=pruner
    )
    
    study.optimize(objective, n_trials=n_trials, timeout=30)
    
    best_params = study.best_params
    best_score = study.best_value
    
    # Get final features (most common across folds)
    all_features = [f for features in fold_features for f in features]
    feature_counts = Counter(all_features)
    final_features = [f for f, count in feature_counts.items() 
                     if count >= len(fold_features) // 2]
    
    if len(final_features) < 10:
        final_features = [f for f, _ in feature_counts.most_common(50)]
    
    return best_params, final_features, best_score


def optimize_logreg_bayesian(X_train, y_train, cv_splits, fold_features, config):
    """Bayesian optimization for Logistic Regression with pre-computed features."""
    bayesian_config = config.bayesian_ranges.LogReg
    n_trials = config.optimization.n_trials
    startup_trials = config.optimization.startup_trials
    
    def objective(trial):
        C = trial.suggest_float('C', bayesian_config.C_range[0], 
                               bayesian_config.C_range[1], log=True)
        penalty = trial.suggest_categorical('penalty', bayesian_config.penalties)
        
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            selected_features = fold_features[fold_idx]
            
            X_fold_train = X_train.iloc[train_idx][selected_features]
            X_fold_val = X_train.iloc[val_idx][selected_features]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            scaler = RobustScaler()
            X_fold_train_scaled = scaler.fit_transform(X_fold_train)
            X_fold_val_scaled = scaler.transform(X_fold_val)
            
            logreg = LogisticRegression(C=C, penalty=penalty, solver='liblinear', 
                                      random_state=GLOBAL_SEED, max_iter=1000)
            logreg.fit(X_fold_train_scaled, y_fold_train)
            
            score = logreg.score(X_fold_val_scaled, y_fold_val)
            fold_scores.append(score)
            
            trial.report(score, fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return float(np.mean(fold_scores))

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1)
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42, n_startup_trials=startup_trials, multivariate=True),
        pruner=pruner
    )
    
    study.optimize(objective, n_trials=n_trials, timeout=30)
    
    best_params = study.best_params
    best_score = study.best_value
    
    # Get final features
    all_features = [f for features in fold_features for f in features]
    feature_counts = Counter(all_features)
    final_features = [f for f, count in feature_counts.items() 
                     if count >= len(fold_features) // 2]
    
    if len(final_features) < 10:
        final_features = [f for f, _ in feature_counts.most_common(50)]
    
    return best_params, final_features, best_score


def optimize_rf_bayesian(X_train, y_train, cv_splits, fold_features, config):
    """Bayesian optimization for Random Forest with pre-computed features."""
    bayesian_config = config.bayesian_ranges.RF
    n_trials = config.optimization.n_trials
    startup_trials = config.optimization.startup_trials
    
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 
                                       bayesian_config.n_estimators_range[0], 
                                       bayesian_config.n_estimators_range[1])
        max_depth = trial.suggest_int('max_depth', 
                                    bayesian_config.max_depth_range[0], 
                                    bayesian_config.max_depth_range[1])
        min_samples_split = trial.suggest_int('min_samples_split', 
                                            bayesian_config.min_samples_split_range[0], 
                                            bayesian_config.min_samples_split_range[1])
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 
                                           bayesian_config.min_samples_leaf_range[0], 
                                           bayesian_config.min_samples_leaf_range[1])
        
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            selected_features = fold_features[fold_idx]
            
            X_fold_train = X_train.iloc[train_idx][selected_features]
            X_fold_val = X_train.iloc[val_idx][selected_features]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            scaler = RobustScaler()
            X_fold_train_scaled = scaler.fit_transform(X_fold_train)
            X_fold_val_scaled = scaler.transform(X_fold_val)
            
            rf = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                random_state=GLOBAL_SEED, n_jobs=-1
            )
            rf.fit(X_fold_train_scaled, y_fold_train)
            
            score = rf.score(X_fold_val_scaled, y_fold_val)
            fold_scores.append(score)
            
            trial.report(score, fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return float(np.mean(fold_scores))
    
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1)
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42, n_startup_trials=startup_trials, multivariate=True),
        pruner=pruner
    )
    
    study.optimize(objective, n_trials=n_trials, timeout=30)
    
    best_params = study.best_params
    best_score = study.best_value
    
    # Get final features
    all_features = [f for features in fold_features for f in features]
    feature_counts = Counter(all_features)
    final_features = [f for f, count in feature_counts.items() 
                     if count >= len(fold_features) // 2]
    
    if len(final_features) < 10:
        final_features = [f for f, _ in feature_counts.most_common(50)]
    
    return best_params, final_features, best_score


def train_final_model(model_type, X_train, y_train, X_test, y_test, 
                     selected_features, best_params):
    """Train and evaluate final model with optimized parameters."""
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    if model_type == 'SVM':
        model = SVC(**best_params, random_state=GLOBAL_SEED)
    elif model_type == 'LogReg':
        model = LogisticRegression(**best_params, solver='liblinear', 
                                 random_state=GLOBAL_SEED, max_iter=1000)
    elif model_type == 'RF':
        model = RandomForestClassifier(**best_params, random_state=GLOBAL_SEED, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train_scaled, y_train)
    
    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)
    
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    return {
        'model': model, 
        'scaler': scaler,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_predictions': train_pred,
        'test_predictions': test_pred
    }


def train_loso_source_models(source_data, feature_cols, target_col, config):
    """Train LOSO models with unified feature selection."""
    X_source = source_data[feature_cols].copy()
    y_source = source_data[target_col].copy()
    
    # Clean data
    X_source.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_source = X_source.fillna(X_source.median())

    # Unified feature selection
    max_features = config.feature_selection.max_mRMR_features
    selected_features = select_features_global(X_source, y_source, max_features)
    X_source_selected = X_source[selected_features]

    # CV setup for optimization
    cv_folds = get_cv_folds(config)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=GLOBAL_SEED)
    cv_splits = list(skf.split(X_source_selected, y_source))
    
    # Pre-compute features for all folds
    fold_features = precompute_fold_features(X_source_selected, y_source, cv_splits, max_features)

    # Optimize models
    svm_params, svm_features, svm_score = optimize_svm_bayesian(
        X_source_selected, y_source, cv_splits, fold_features, config)
    
    logreg_params, logreg_features, logreg_score = optimize_logreg_bayesian(
        X_source_selected, y_source, cv_splits, fold_features, config)
    
    rf_params, rf_features, rf_score = optimize_rf_bayesian(
        X_source_selected, y_source, cv_splits, fold_features, config)

    # Train final models on full source data
    final_scaler = RobustScaler().fit(X_source_selected)
    X_source_final = final_scaler.transform(X_source_selected)

    final_svm = SVC(**svm_params, probability=True, random_state=GLOBAL_SEED)
    final_svm.fit(X_source_final, y_source)
    
    final_logreg = LogisticRegression(**logreg_params, solver='liblinear', 
                                    random_state=GLOBAL_SEED, max_iter=1000)
    final_logreg.fit(X_source_final, y_source)
    
    final_rf = RandomForestClassifier(**rf_params, random_state=GLOBAL_SEED, n_jobs=-1)
    final_rf.fit(X_source_final, y_source)

    trained_models = {
        'SVM': {'model': final_svm, 'scaler': final_scaler, 'best_params': svm_params, 'cv_score': svm_score},
        'LogReg': {'model': final_logreg, 'scaler': final_scaler, 'best_params': logreg_params, 'cv_score': logreg_score},
        'RF': {'model': final_rf, 'scaler': final_scaler, 'best_params': rf_params, 'cv_score': rf_score}
    }
    
    optimization_info = {
        'SVM': {'best_cv_score': svm_score, 'best_params': svm_params},
        'LogReg': {'best_cv_score': logreg_score, 'best_params': logreg_params},
        'RF': {'best_cv_score': rf_score, 'best_params': rf_params}
    }

    return trained_models, selected_features, optimization_info


def evaluate_zero_shot_models(models, evaluation_data, selected_features, target_col):
    """Evaluate 0-shot performance of source models on target data."""
    results = {}
    y_eval = evaluation_data[target_col]
    
    # Ensure all features are present
    X_eval_raw = evaluation_data.copy()
    for f in selected_features:
        if f not in X_eval_raw.columns:
            X_eval_raw[f] = 0
    X_eval = X_eval_raw[selected_features]

    for model_name, model_info in models.items():
        try:
            scaler = model_info['scaler']
            X_eval_scaled = scaler.transform(X_eval)
            
            predictions = model_info['model'].predict(X_eval_scaled)
            accuracy = accuracy_score(y_eval, predictions)
            
            results[model_name] = {
                'accuracy': accuracy,
                'source_cv_score': model_info.get('cv_score', 0.0)
            }
        except Exception as e:
            print(f"Error evaluating {model_name} (0-shot): {e}")
            results[model_name] = {'accuracy': 0.0, 'error': str(e)}
            
    return results


def calibrate_and_evaluate_models(source_models, calibration_data, evaluation_data, 
                                 selected_features, target_col):
    """Calibrate source models on target data and evaluate."""
    calibrated_results = {}
    y_cal = calibration_data[target_col]
    y_eval = evaluation_data[target_col]

    X_cal = calibration_data[selected_features]
    X_eval = evaluation_data[selected_features]

    for model_name, model_info in source_models.items():
        try:
            scaler = model_info['scaler']
            X_cal_scaled = scaler.transform(X_cal)
            X_eval_scaled = scaler.transform(X_eval)

            # Create new model with same hyperparameters
            calibrated_model = clone(model_info['model'])
            calibrated_model.fit(X_cal_scaled, y_cal)

            predictions = calibrated_model.predict(X_eval_scaled)
            accuracy = accuracy_score(y_eval, predictions)
            
            calibrated_results[model_name] = {'accuracy': accuracy}

        except Exception as e:
            print(f"Error calibrating {model_name}: {e}")
            calibrated_results[model_name] = {'accuracy': 0.0, 'error': str(e)}
            
    return calibrated_results


def run_cross_validation_analysis(data, subject_id, config, analysis_params):
    """Main function for cross-validation analysis with Bayesian optimization."""
    from datetime import datetime
    import time
    import pickle
    import os
    
    model_savepath = config.paths.model_savepath
    n_repetitions = config.cross_validation.n_repetitions
    k_out = config.cross_validation.k_out
    
    n_models = 3
    target_col = "Condition"
    non_feature_cols = ["Trial_index", "Condition_name", "Label", "Condition"]
    feature_cols = [col for col in data.columns if col not in non_feature_cols]
    
    # Check class distribution
    class_counts = data[target_col].value_counts()
    print(f"Class distribution: {class_counts.to_dict()}")

    # Handle shuffling
    if analysis_params.get('shuffle', False) and not analysis_params.get('time_based', False):
        data[target_col] = data[target_col].sample(frac=1, random_state=GLOBAL_SEED).reset_index(drop=True)
        print("Labels shuffled.")

    # Initialize arrays
    accuracy_arr = np.zeros((n_repetitions, k_out, n_models, 3))
    model_parameters = []
    final_models = []
    final_features = []
    final_predictions = []

    # Prepare cross-validation splits
    if analysis_params.get('time_based', False):
        n_trials = data.shape[0]
        n_train_trials = int(n_trials * 0.8)
        train_indices = np.arange(n_train_trials)
        test_indices = np.arange(n_train_trials, n_trials)
        outer_cv = [(train_indices, test_indices)]
        n_repetitions = 1
        k_out = 1
        print("Using time-based CV split.")
    else:
        outer_cv = []
        for rep in range(n_repetitions):
            skf = StratifiedKFold(n_splits=k_out, shuffle=True, random_state=GLOBAL_SEED + rep)
            splits = list(skf.split(data[feature_cols], data[target_col]))
            outer_cv.append(splits)

    # Start analysis
    start_time = datetime.now()
    total_trials = 0

    for rep in range(n_repetitions):
        print(f"Outer fold repetition {rep + 1}/{n_repetitions}")
        
        cv_rep = outer_cv[rep] if not analysis_params.get('time_based', False) else outer_cv
        model_params_rep = []
        models_rep = []
        features_rep = []
        predictions_rep = []
        
        for fold_idx, (train_index, test_index) in enumerate(cv_rep):
            print(f"Processing outer fold {fold_idx + 1}/{k_out}")
            fold_start_time = time.time()
            
            # Prepare data
            X_train_raw = data.iloc[train_index][feature_cols]
            X_test_raw = data.iloc[test_index][feature_cols]
            y_train = data.iloc[train_index][target_col]
            y_test = data.iloc[test_index][target_col]
            
            # Reset indices
            X_train_raw = X_train_raw.reset_index(drop=True)
            X_test_raw = X_test_raw.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)

            print(f"Fold {fold_idx + 1}: Train={len(X_train_raw)}, Test={len(X_test_raw)}")

            # Create inner CV splits
            if analysis_params.get('time_based', False):
                n_train_samples = X_train_raw.shape[0]
                n_inner_train = int(n_train_samples * 0.8)
                inner_train_indices = np.arange(n_inner_train)
                inner_val_indices = np.arange(n_inner_train, n_train_samples)
                inner_cv_splits = [(inner_train_indices, inner_val_indices)]
            else:
                inner_cv_splits = create_inner_cv_splits(
                    X_train_raw, y_train, 3, rep, GLOBAL_SEED
                )

            print(f"Inner CV: {len(inner_cv_splits)} folds")
            
            # Global feature selection for consistency
            max_features = config.feature_selection.max_mRMR_features
            global_features = select_features_global(X_train_raw, y_train, max_features)
            fold_features = [global_features] * len(inner_cv_splits)
            
            print("Starting Bayesian optimization...")
            
            # Optimize models
            svm_params, svm_features, svm_score = optimize_svm_bayesian(
                X_train_raw, y_train, inner_cv_splits, fold_features, config)
            
            logreg_params, logreg_features, logreg_score = optimize_logreg_bayesian(
                X_train_raw, y_train, inner_cv_splits, fold_features, config)

            rf_params, rf_features, rf_score = optimize_rf_bayesian(
                X_train_raw, y_train, inner_cv_splits, fold_features, config)

            total_trials += config.optimization.n_trials * 3
            
            print("Training final models...")
            
            # Train final models
            svm_result = train_final_model('SVM', X_train_raw, y_train, X_test_raw, y_test, 
                                         svm_features, svm_params)
            logreg_result = train_final_model('LogReg', X_train_raw, y_train, X_test_raw, y_test,
                                            logreg_features, logreg_params)
            rf_result = train_final_model('RF', X_train_raw, y_train, X_test_raw, y_test,
                                        rf_features, rf_params)

            fold_time = time.time() - fold_start_time
            print(f"Fold {fold_idx + 1} completed in {fold_time:.1f}s:")
            print(f"  SVM: Train={svm_result['train_accuracy']:.3f}, Val={svm_score:.3f}, Test={svm_result['test_accuracy']:.3f}")
            print(f"  LogReg: Train={logreg_result['train_accuracy']:.3f}, Val={logreg_score:.3f}, Test={logreg_result['test_accuracy']:.3f}")
            print(f"  RF: Train={rf_result['train_accuracy']:.3f}, Val={rf_score:.3f}, Test={rf_result['test_accuracy']:.3f}")

            # Store results
            accuracy_arr[rep, fold_idx, :, :] = [
                [svm_result['train_accuracy'], svm_score, svm_result['test_accuracy']],
                [logreg_result['train_accuracy'], logreg_score, logreg_result['test_accuracy']],
                [rf_result['train_accuracy'], rf_score, rf_result['test_accuracy']]
            ]
            
            # Store parameters and models
            svm_params_list = [svm_params['C'], svm_params['kernel'], svm_params.get('gamma', 'scale'), len(svm_features)]
            logreg_params_list = [logreg_params['C'], logreg_params['penalty'], len(logreg_features)]
            rf_params_list = [rf_params['n_estimators'], rf_params['max_depth'], 
                            rf_params['min_samples_split'], rf_params['min_samples_leaf'], len(rf_features)]
            
            model_params_rep.append([svm_params_list, logreg_params_list, rf_params_list])
        
            models_rep.append([
                {'model': svm_result['model'], 'scaler': svm_result['scaler'], 'features': svm_features, 'X_test': X_test_raw, 'y_test': y_test},
                {'model': logreg_result['model'], 'scaler': logreg_result['scaler'], 'features': logreg_features, 'X_test': X_test_raw, 'y_test': y_test},
                {'model': rf_result['model'], 'scaler': rf_result['scaler'], 'features': rf_features, 'X_test': X_test_raw, 'y_test': y_test}
            ])
            
            features_rep.extend([
                ['SVM', svm_features, np.ones(len(svm_features)) / len(svm_features)],
                ['LogReg', logreg_features, np.ones(len(logreg_features)) / len(logreg_features)],
                ['RF', rf_features, np.ones(len(rf_features)) / len(rf_features)]
            ])
            
            predictions_rep.append([
                [svm_result['train_predictions'], svm_result['test_predictions']],
                [logreg_result['train_predictions'], logreg_result['test_predictions']],
                [rf_result['train_predictions'], rf_result['test_predictions']]
            ])
            
        model_parameters.append(model_params_rep)
        final_models.append(models_rep)
        final_features.append(features_rep)
        final_predictions.append(predictions_rep)

    # Save results
    results = [accuracy_arr, model_parameters, final_models, final_features, final_predictions]

    condition_strings = []
    if analysis_params.get('condition_filter'):
        condition_strings.append(analysis_params['condition_filter'].lower())
    if analysis_params.get('shuffle', False):
        condition_strings.append('shuffled')
    if analysis_params.get('time_based', False):
        condition_strings.append('time_based')

    condition_suffix = '_'.join(condition_strings) if condition_strings else 'all_conditions'
    save_filename = f"Sub_{subject_id}_{condition_suffix}_results.pkl"
    save_path = os.path.join(model_savepath, save_filename)

    with open(save_path, "wb") as filehandle:
        pickle.dump(results, filehandle)

    end_time = datetime.now()
    time_difference = end_time - start_time
    print(f"Analysis completed in {time_difference}")
    print(f"Total optimization trials: {total_trials}")
    print(f"Results saved to: {save_path}")
    
    # Print summary
    print(f"Summary for {subject_id}:")
    mean_accuracies = np.nanmean(accuracy_arr, axis=(0, 1))
    model_names = ['SVM', 'LogReg', 'RF']
    acc_types = ['Train', 'Val', 'Test']
    
    for i, model_name in enumerate(model_names):
        for j, acc_type in enumerate(acc_types):
            print(f"{model_name} {acc_type} Accuracy: {mean_accuracies[i, j]:.3f}")

    return results


def run_loso_transfer_learning(features_path, config, condition_filter=None, analysis_params=None):
    """Run LOSO transfer learning with unified feature set and calibration."""
    if analysis_params is None: 
        analysis_params = {}
        
    subjects = get_available_subjects(features_path)
    all_subject_data = {}
    
    for s in subjects:
        data = load_subject_data(s, features_path, condition_filter)
        if data is not None:
            all_subject_data[s] = data
    
    if len(all_subject_data) < 3:
        print(f"Need at least 3 subjects for LOSO, found {len(all_subject_data)}")
        return None

    non_feature_cols = ["Trial_index", "Condition_name", "Label", "Condition"]
    common_features = sorted(list(set.intersection(
        *(set(df.columns) - set(non_feature_cols) for df in all_subject_data.values())
    )))
    print(f"Found {len(common_features)} common features across all subjects")

    results = {
        'zero_shot_results': {}, 
        'calibrated_results': {}, 
        'model_parameters': {}, 
        'feature_selection_info': {}, 
        'bayesian_optimization_info': {}
    }

    for target_subject in subjects:
        if target_subject not in all_subject_data: 
            continue
            
        print(f"Target subject: {target_subject}")
        
        source_subjects = [s for s in subjects if s != target_subject and s in all_subject_data]
        combined_source_data = pd.concat([all_subject_data[s] for s in source_subjects], ignore_index=True)
        target_data = all_subject_data[target_subject]
        target_col = "Condition"

        # Train models on source data
        trained_models, selected_features, bayesian_info = train_loso_source_models(
            combined_source_data, common_features, target_col, config)
        
        results['bayesian_optimization_info'][target_subject] = bayesian_info
        results['feature_selection_info'][target_subject] = {
            'selected_features_count': len(selected_features), 
            'common_features_count': len(common_features), 
            'selected_features': selected_features
        }

        # Split target data for calibration and evaluation
        n_calibration_trials = min(30, int(len(target_data) * 0.25))
        if len(target_data) - n_calibration_trials < 10: 
            n_calibration_trials = 10

        calibration_data, evaluation_data = train_test_split(
            target_data, train_size=n_calibration_trials, 
            stratify=target_data[target_col], random_state=GLOBAL_SEED)

        # Evaluate 0-shot and calibrated performance
        print("Evaluating 0-shot performance...")
        zero_shot_res = evaluate_zero_shot_models(trained_models, evaluation_data, selected_features, target_col)
        
        print("Calibrating models and evaluating...")
        calibrated_res = calibrate_and_evaluate_models(trained_models, calibration_data, evaluation_data, selected_features, target_col)

        results['zero_shot_results'][target_subject] = zero_shot_res
        results['calibrated_results'][target_subject] = calibrated_res
        results['model_parameters'][target_subject] = {
            'n_source_trials': len(combined_source_data), 
            'n_calibration_trials': len(calibration_data), 
            'n_evaluation_trials': len(evaluation_data)
        }

    # Save results
    condition_suffix_parts = []
    if condition_filter:
        condition_suffix_parts.append(condition_filter.lower())
    else:
        condition_suffix_parts.append("all_conditions")

    if analysis_params.get('shuffle', False):
        condition_suffix_parts.append('shuffled')
    if analysis_params.get('time_based', False):
        condition_suffix_parts.append('temporal')

    condition_suffix = '_'.join(condition_suffix_parts)
    save_filename = f"LOSO_results_{condition_suffix}.pkl"
    save_path = os.path.join(config.paths.model_savepath, save_filename)

    with open(save_path, "wb") as f:
        pickle.dump(results, f)

    print(f"LOSO analysis complete. Results saved to: {save_path}")
    return results