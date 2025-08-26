#%%!/usr/bin/env python3
"""
Debug script to inspect the structure of your saved data files.
"""

import numpy as np
from pathlib import Path

def inspect_data_files(features_path, subject_id):
    """
    Inspect the structure of your data files to understand the format.
    """
    features_path = Path(features_path)
    print(f"Inspecting data for subject: {subject_id}")
    print("="*60)
    
    # Check binary labels file
    labels_file = features_path / f"{subject_id}_binary_labels.npz"
    if labels_file.exists():
        print(f"\n📁 Binary Labels File: {labels_file.name}")
        try:
            labels_data = np.load(labels_file, allow_pickle=True)
            print(f"Keys: {list(labels_data.keys())}")
            for key in labels_data.keys():
                data = labels_data[key]
                print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
                if hasattr(data, '__len__') and len(data) < 20:
                    print(f"    Sample values: {data}")
                elif hasattr(data, '__len__'):
                    print(f"    Sample values: {data[:5]}... (showing first 5)")
        except Exception as e:
            print(f"  Error loading: {e}")
    else:
        print(f"❌ Binary labels file not found: {labels_file}")
    
    # Check trial info file
    trial_info_file = features_path / f"{subject_id}_trial_info.npz"
    if trial_info_file.exists():
        print(f"\n📁 Trial Info File: {trial_info_file.name}")
        try:
            trial_data = np.load(trial_info_file, allow_pickle=True)
            print(f"Keys: {list(trial_data.keys())}")
            for key in trial_data.keys():
                data = trial_data[key]
                print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
                if key == 'conditions' and hasattr(data, '__len__'):
                    unique_conditions = np.unique(data)
                    print(f"    Unique conditions: {unique_conditions}")
                    condition_counts = np.unique(data, return_counts=True)
                    print(f"    Condition counts: {dict(zip(condition_counts[0], condition_counts[1]))}")
                elif hasattr(data, '__len__') and len(data) < 20:
                    print(f"    Sample values: {data}")
                elif hasattr(data, '__len__'):
                    print(f"    Sample values: {data[:5]}... (showing first 5)")
        except Exception as e:
            print(f"  Error loading: {e}")
    else:
        print(f"❌ Trial info file not found: {trial_info_file}")
    
    # Check feature files
    feature_files = {
        'psd': f"{subject_id}_psd_features.npz",
        'wpli': f"{subject_id}_hjorth_wpli_features.npz", 
        'pac': f"{subject_id}_hjorth_pac_features.npz",
        'phase': f"{subject_id}_hjorth_phase_features.npz"
    }
    
    for feature_type, filename in feature_files.items():
        filepath = features_path / filename
        if filepath.exists():
            print(f"\n📁 {feature_type.upper()} Features File: {filename}")
            try:
                feature_data = np.load(filepath, allow_pickle=True)
                print(f"Keys: {list(feature_data.keys())}")
                for key in feature_data.keys():
                    data = feature_data[key]
                    print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
                    # Check for NaN values
                    if np.issubdtype(data.dtype, np.floating):
                        nan_count = np.sum(np.isnan(data))
                        print(f"    NaN count: {nan_count}")
            except Exception as e:
                print(f"  Error loading: {e}")
        else:
            print(f"❌ {feature_type.upper()} features file not found: {filename}")

def test_condition_filtering():
    """
    Test the condition filtering logic.
    """
    features_path = "/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/SICISICF/features_extracted"
    
    # Get first available subject
    features_path_obj = Path(features_path)
    subject_files = list(features_path_obj.glob("*_binary_labels.npz"))
    
    if not subject_files:
        print("No subject files found!")
        return
    
    subject_id = subject_files[0].name.replace("_binary_labels.npz", "")
    print(f"Testing with subject: {subject_id}")
    
    # Inspect the data structure
    inspect_data_files(features_path, subject_id)
    
    # Test loading with different conditions
    print(f"\n{'='*60}")
    print("TESTING CONDITION FILTERING")
    print(f"{'='*60}")
    
    try:
        # Load trial info to see available conditions
        trial_info_file = features_path_obj / f"{subject_id}_trial_info.npz"
        trial_data = np.load(trial_info_file, allow_pickle=True)
        conditions = trial_data['conditions']
        
        print(f"Available conditions in trial data: {np.unique(conditions)}")
        
        # Test filtering for each condition
        for condition in ['SICI', 'SICF', 'Single']:
            condition_mask = conditions == condition
            count = np.sum(condition_mask)
            print(f"{condition}: {count} trials")
            
            if count > 0:
                # Load labels to see how many have valid labels
                labels_file = features_path_obj / f"{subject_id}_binary_labels.npz"
                labels_data = np.load(labels_file, allow_pickle=True)
                valid_mask = labels_data['valid_mask']
                
                # Apply condition filter
                condition_valid_mask = valid_mask[condition_mask]
                valid_count = np.sum(condition_valid_mask)
                
                print(f"  {condition}: {valid_count} trials with valid labels")
                
                if valid_count > 0:
                    labels = labels_data['labels']
                    condition_labels = labels[condition_mask]
                    valid_labels = condition_labels[condition_valid_mask]
                    
                    # Count label distribution
                    unique_labels, label_counts = np.unique(valid_labels[~np.isnan(valid_labels)], return_counts=True)
                    print(f"  {condition}: Label distribution: {dict(zip(unique_labels.astype(int), label_counts))}")
    
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_condition_filtering()
# %%
