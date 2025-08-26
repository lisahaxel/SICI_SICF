#%%!/usr/bin/env python3
"""
Test script to verify the data loading works correctly.
"""

import sys
import os
sys.path.append('src')

from src.helper_functions import load_new_features_df, get_available_subjects

def test_loading():
    """Test loading data for each condition."""
    features_path = "/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/SICISICF/features_extracted"
    
    # Get available subjects
    subjects = get_available_subjects(features_path)
    print(f"Found subjects: {subjects}")
    
    if not subjects:
        print("No subjects found!")
        return
    
    test_subject = subjects[0]
    print(f"\nTesting with subject: {test_subject}")
    
    # Test loading for each condition
    conditions = ['SICI', 'SICF', 'Single']
    
    for condition in conditions:
        print(f"\n{'='*50}")
        print(f"Testing condition: {condition}")
        print(f"{'='*50}")
        
        try:
            df = load_new_features_df(features_path, test_subject, condition)
            
            print(f"✓ Successfully loaded {condition} data")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {df.columns.tolist()[:10]}..." if len(df.columns) > 10 else f"  Columns: {df.columns.tolist()}")
            print(f"  Condition distribution: {df['Condition'].value_counts().to_dict()}")
            
            # Check for any NaN values in features
            feature_cols = [col for col in df.columns if col not in ['Trial_index', 'Condition_name', 'Label', 'Condition']]
            nan_counts = df[feature_cols].isna().sum().sum()
            print(f"  Total NaN values in features: {nan_counts}")
            
            # Show some basic stats
            print(f"  Feature columns: {len(feature_cols)}")
            
            # Count features by type
            feature_types = {}
            for col in feature_cols:
                ftype = col.split('_')[0]
                feature_types[ftype] = feature_types.get(ftype, 0) + 1
            
            print(f"  Features by type: {feature_types}")
            
        except Exception as e:
            print(f"✗ Error loading {condition}: {e}")
            import traceback
            traceback.print_exc()

def test_all_conditions():
    """Test loading all conditions together."""
    features_path = "/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/SICISICF/features_extracted"
    subjects = get_available_subjects(features_path)
    
    if not subjects:
        print("No subjects found!")
        return
    
    test_subject = subjects[0]
    print(f"\n{'='*50}")
    print(f"Testing ALL conditions together")
    print(f"{'='*50}")
    
    try:
        df = load_new_features_df(features_path, test_subject, condition_filter=None)
        
        print(f"✓ Successfully loaded all conditions")
        print(f"  Shape: {df.shape}")
        print(f"  Condition distribution: {df['Condition'].value_counts().to_dict()}")
        print(f"  Condition names: {df['Condition_name'].value_counts().to_dict()}")
        
    except Exception as e:
        print(f"✗ Error loading all conditions: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loading()
    test_all_conditions()
# %%
