
# %%
# Import necessary libraries
import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import scipy.signal
from scipy.stats import zscore
import pactools
from scipy.signal import butter, filtfilt, hilbert
from spectrum import aryule
import argparse
import sys


# Set up MNE to reduce verbosity
mne.set_log_level('WARNING')

def get_config_for_subject(subject_id):
    """Get configuration for a specific subject."""
    USER = os.environ.get('USER', 'user')
    REPO_DIR = Path(f"/mnt/lustre/work/macke/{USER}/repos/eegjepa")
    
    PROCESSED_DATA_PATH = REPO_DIR / "EDAPT_neurips/EDAPT_TMS/SICISICF/data_processed_final_pre_ica_True_final_offline"
    FEATURES_OUTPUT_PATH = REPO_DIR / "EDAPT_neurips/EDAPT_TMS/SICISICF/features_extracted"
    
    # Create output directory if it doesn't exist
    FEATURES_OUTPUT_PATH.mkdir(exist_ok=True)
    
    return PROCESSED_DATA_PATH, FEATURES_OUTPUT_PATH, subject_id

# %%
def load_eeg_and_mep_data(subject_id: str, base_path: Path):
    """
    Loads the preprocessed pre-stimulus EEG epochs and MEP data for a given subject.
    """
    subject_path = base_path / subject_id
    eeg_file_path = subject_path / f"{subject_id}_pre-epo.fif"
    mep_file_path = subject_path / f"{subject_id}_MEPs.npy"

    if not eeg_file_path.exists() or not mep_file_path.exists():
        print(f"Error: Data files not found for subject {subject_id}")
        return None, None

    print(f"Loading data for subject: {subject_id}")
    try:
        epochs_pre = mne.read_epochs(eeg_file_path, preload=True, verbose=False)
        meps = np.load(mep_file_path)
        print(f"Data loaded successfully. Epochs shape: {epochs_pre.get_data().shape}, MEPs shape: {meps.shape}")
        return epochs_pre, meps
    except Exception as e:
        print(f"Error loading files: {e}")
        return None, None
    
def extract_trial_info_and_ratios(epochs, meps, subject_id, output_path):
    """
    Extract trial information, calculate SICI/SICF ratios, and create visualization.
    Clean, consolidated function to handle all trial-related processing.
    """
    print("\n=== Extracting Trial Information and Calculating Ratios ===")
    
    # Fix MEPs shape if needed
    if meps.ndim == 2 and meps.shape[0] == 1:
        meps = meps.squeeze(0)  # Remove singleton dimension: (1, 1173) -> (1173,)
    print(f"MEPs shape after squeeze: {meps.shape}")
    
    # Extract basic trial information
    trial_indices = epochs.events[:, 0]  # Sample indices of trials
    
    # Convert event codes to condition names
    conditions_ordered = np.array([None] * len(epochs))
    
    # Create proper mapping: 1->Single, 2->SICI, 3->SICF
    event_code_mapping = {1: 'Single', 2: 'SICI', 3: 'SICF'}
    
    for i, event_code in enumerate(epochs.events[:, 2]):
        if event_code in event_code_mapping:
            conditions_ordered[i] = event_code_mapping[event_code]
        else:
            print(f"Warning: Unknown event code {event_code} at trial {i}")
            conditions_ordered[i] = f"Unknown_{event_code}"
    
    print(f"Condition distribution: {np.unique(conditions_ordered, return_counts=True)}")
    
    # Calculate log-transformed MEPs
    meps = meps * 1e6  # Convert from Volts to microvolts
    mep_log_transformed = np.log1p(np.abs(meps) + 1e-12)
    
    # Identify blocks based on transitions TO Single pulse trials
    def identify_blocks(conditions):
        blocks = []
        current_block = []
        
        for i, condition in enumerate(conditions):
            # Start a new block when transitioning from non-Single to Single
            if condition == 'Single' and i > 0 and conditions[i-1] != 'Single':
                # Finish the current block if it exists
                if len(current_block) > 0:
                    blocks.append(current_block)
                # Start new block with this Single trial
                current_block = [i]
            else:
                # Add trial to current block
                current_block.append(i)
        
        # Add the last block
        if len(current_block) > 0:
            blocks.append(current_block)
        
        return blocks
    
    # Get block assignments
    blocks = identify_blocks(conditions_ordered)
    block_assignments = np.full(len(conditions_ordered), -1)
    
    for block_num, trial_indices_in_block in enumerate(blocks):
        for trial_idx in trial_indices_in_block:
            block_assignments[trial_idx] = block_num
    
    # Initialize arrays for ratios
    mep_ratios = np.full(len(meps), np.nan)
    
    # Calculate ratios for each block
    for block_num, trial_indices_in_block in enumerate(blocks):
        block_conditions = conditions_ordered[trial_indices_in_block]
        block_meps_log = mep_log_transformed[trial_indices_in_block]
        
        # Find ALL single-pulse trials in this block
        sp_mask = block_conditions == 'Single'
        
        if np.any(sp_mask):
            # Calculate mean of ALL log-transformed single-pulse MEPs for this block
            block_sp_mean = np.mean(block_meps_log[sp_mask])
            
            # Calculate ratios for SICI and SICF trials in this block
            for local_idx, global_idx in enumerate(trial_indices_in_block):
                if block_conditions[local_idx] in ['SICI', 'SICF']:
                    # Ratio = current trial MEP / mean of single-pulse MEPs in this block
                    mep_ratios[global_idx] = block_meps_log[local_idx] / block_sp_mean
        else:
            print(f"Warning: No single-pulse trials found in block {block_num}")
    
    # Create comprehensive trial info
    trial_info = {
        'trial_indices': trial_indices,
        'conditions': conditions_ordered,
        'mep_raw': meps,
        'mep_log_transformed': mep_log_transformed,
        'mep_ratios': mep_ratios,
        'block_assignments': block_assignments
    }
    
    # Save trial information
    trial_info_file = output_path / f"{subject_id}_trial_info.npz"
    np.savez(trial_info_file, **trial_info)
    print(f"Trial information saved to: {trial_info_file}")
    
    # Create MEP visualization
    print("Creating MEP visualization...")
    create_mep_visualization(trial_info, subject_id, output_path)
    
    # Print summary statistics
    print_trial_summary(trial_info, blocks)
    
    return trial_info

def create_mep_visualization(trial_info, subject_id, output_path):
    """Create visualization of MEP data by condition."""
    import matplotlib.pyplot as plt
    
    conditions_ordered = trial_info['conditions']
    meps = trial_info['mep_raw']
    mep_log_transformed = trial_info['mep_log_transformed']
    mep_ratios = trial_info['mep_ratios']
    
    plt.figure(figsize=(15, 10))
    
    # Define colors for conditions
    colors = {'Single': 'blue', 'SICI': 'red', 'SICF': 'green'}
    markers = {'Single': 'o', 'SICI': 's', 'SICF': '^'}
    
    # Plot raw MEPs
    plt.subplot(3, 1, 1)
    for condition in ['Single', 'SICI', 'SICF']:
        mask = conditions_ordered == condition
        if np.any(mask):
            trial_nums = np.where(mask)[0] + 1  # 1-indexed trial numbers
            plt.scatter(trial_nums, meps[mask], 
                       c=colors[condition], marker=markers[condition], 
                       alpha=0.7, s=50, label=f'{condition} (n={np.sum(mask)})')
    
    plt.xlabel('Trial Number')
    plt.ylabel('Raw MEP Amplitude (μV)')
    plt.title(f'{subject_id} - Raw MEP Amplitudes by Condition')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot log-transformed MEPs
    plt.subplot(3, 1, 2)
    for condition in ['Single', 'SICI', 'SICF']:
        mask = conditions_ordered == condition
        if np.any(mask):
            trial_nums = np.where(mask)[0] + 1  # 1-indexed trial numbers
            plt.scatter(trial_nums, mep_log_transformed[mask], 
                       c=colors[condition], marker=markers[condition], 
                       alpha=1, s=50, label=f'{condition} (n={np.sum(mask)})')
    
    plt.xlabel('Trial number')
    plt.ylabel('Log-MEP amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot ratios (only for SICI and SICF)
    plt.subplot(3, 1, 3)
    for condition in ['SICI', 'SICF']:
        mask = (conditions_ordered == condition) & (~np.isnan(mep_ratios))
        if np.any(mask):
            trial_nums = np.where(conditions_ordered == condition)[0] + 1
            ratios_condition = mep_ratios[conditions_ordered == condition]
            # Only plot non-NaN ratios
            valid_mask = ~np.isnan(ratios_condition)
            if np.any(valid_mask):
                plt.scatter(trial_nums[valid_mask], ratios_condition[valid_mask], 
                           c=colors[condition], marker=markers[condition], 
                           alpha=1, s=50, label=f'{condition} (n={np.sum(valid_mask)})')
    
    plt.xlabel('Trial Number')
    plt.ylabel('MEP Ratio (CR/UR)')
    plt.title(f'{subject_id} - SICI/SICF Ratios by Condition')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = output_path / f"{subject_id}_mep_by_condition.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"MEP visualization saved to: {plot_filename}")

def print_trial_summary(trial_info, blocks):
    """Print comprehensive summary of trial information."""
    conditions_ordered = trial_info['conditions']
    meps = trial_info['mep_raw']
    mep_log_transformed = trial_info['mep_log_transformed']
    mep_ratios = trial_info['mep_ratios']
    
    print(f"\nTrial and Block Summary:")
    print(f"Total trials: {len(conditions_ordered)}")
    print(f"Number of blocks identified: {len(blocks)}")
    
    for i, block in enumerate(blocks):
        block_conds = conditions_ordered[block]
        unique_conds, counts = np.unique(block_conds, return_counts=True)
        cond_str = ", ".join([f"{cond}: {count}" for cond, count in zip(unique_conds, counts)])
        print(f"  Block {i}: {len(block)} trials ({cond_str})")
    
    print(f"\nCondition Statistics:")
    for condition in ['Single', 'SICI', 'SICF']:
        mask = conditions_ordered == condition
        if np.any(mask):
            raw_vals = meps[mask]
            log_vals = mep_log_transformed[mask]
            print(f"{condition} (n={np.sum(mask)}):")
            print(f"  Raw MEP - Mean: {np.mean(raw_vals):.2f} μV, Std: {np.std(raw_vals):.2f} μV")
            print(f"  Log MEP - Mean: {np.mean(log_vals):.3f}, Std: {np.std(log_vals):.3f}")
            
            if condition in ['SICI', 'SICF']:
                ratio_vals = mep_ratios[mask]
                valid_ratios = ratio_vals[~np.isnan(ratio_vals)]
                if len(valid_ratios) > 0:
                    print(f"  Ratios - Mean: {np.mean(valid_ratios):.3f}, Std: {np.std(valid_ratios):.3f}")
                    print(f"  Ratios available: {len(valid_ratios)}/{len(ratio_vals)}")
                else:
                    print(f"  No valid ratios calculated")


def create_median_split_labels(trial_info):
    """
    Create binary labels using median split for each condition separately.
    """
    print(f"\n=== Creating Binary Labels using Median Split ===")
    
    conditions = trial_info['conditions']
    mep_ratios = trial_info['mep_ratios']
    mep_log = trial_info['mep_log_transformed']
    
    labels = np.full(len(conditions), np.nan)
    label_info = {}
    
    # SICI: Median split on ratios
    sici_mask = conditions == 'SICI'
    if np.any(sici_mask):
        sici_ratios = mep_ratios[sici_mask]
        valid_sici = sici_ratios[~np.isnan(sici_ratios)]
        
        if len(valid_sici) > 0:
            median_threshold = np.median(valid_sici)
            # Below median = label 1, above median = label 0
            labels[sici_mask] = (sici_ratios < median_threshold).astype(float)
            
            label_info['SICI'] = {
                'median_threshold': median_threshold,
                'n_valid': len(valid_sici),
                'n_below_median': np.sum(labels[sici_mask] == 1),
                'n_above_median': np.sum(labels[sici_mask] == 0),
                'interpretation': 'Label 1 = Below median (better inhibition), Label 0 = Above median (worse inhibition)'
            }
            print(f"SICI: median={median_threshold:.3f}, below median={label_info['SICI']['n_below_median']}, above median={label_info['SICI']['n_above_median']}")
    
    # SICF: Median split on ratios  
    sicf_mask = conditions == 'SICF'
    if np.any(sicf_mask):
        sicf_ratios = mep_ratios[sicf_mask]
        valid_sicf = sicf_ratios[~np.isnan(sicf_ratios)]
        
        if len(valid_sicf) > 0:
            median_threshold = np.median(valid_sicf)
            # Above median = label 1, below median = label 0
            labels[sicf_mask] = (sicf_ratios > median_threshold).astype(float)
            
            label_info['SICF'] = {
                'median_threshold': median_threshold,
                'n_valid': len(valid_sicf),
                'n_above_median': np.sum(labels[sicf_mask] == 1),
                'n_below_median': np.sum(labels[sicf_mask] == 0),
                'interpretation': 'Label 1 = Above median (better facilitation), Label 0 = Below median (worse facilitation)'
            }
            print(f"SICF: median={median_threshold:.3f}, above median={label_info['SICF']['n_above_median']}, below median={label_info['SICF']['n_below_median']}")
    
    # Single: Median split on log-transformed MEPs
    single_mask = conditions == 'Single'
    if np.any(single_mask):
        single_meps = mep_log[single_mask]
        median_threshold = np.median(single_meps)
        
        # Above median = label 1, below median = label 0
        labels[single_mask] = (single_meps > median_threshold).astype(float)
        
        label_info['Single'] = {
            'median_threshold': median_threshold,
            'n_valid': len(single_meps),
            'n_above_median': np.sum(labels[single_mask] == 1),
            'n_below_median': np.sum(labels[single_mask] == 0),
            'interpretation': 'Label 1 = Above median (higher amplitude), Label 0 = Below median (lower amplitude)'
        }
        print(f"Single: median={median_threshold:.3f}, above median={label_info['Single']['n_above_median']}, below median={label_info['Single']['n_below_median']}")
    
    # Remove NaN labels
    valid_label_mask = ~np.isnan(labels)
    labels_clean = labels[valid_label_mask].astype(int)
    
    # Create label results
    label_results = {
        'labels': labels,  # Full array with NaNs
        'labels_clean': labels_clean,  # Only valid labels
        'valid_mask': valid_label_mask,
        'label_info': label_info
    }
    
    print(f"\nLabel Summary:")
    print(f"Total trials: {len(labels)}")
    print(f"Valid labels: {len(labels_clean)}")
    print(f"Overall distribution: {np.bincount(labels_clean)} (Label 0, Label 1)")
    
    return label_results

def save_labels_simple(trial_info, label_results, subject_id, output_path):
    """Save binary labels with simple summary."""
    
    # Save labels
    labels_file = output_path / f"{subject_id}_binary_labels.npz"
    np.savez(labels_file, 
             labels=label_results['labels'],
             labels_clean=label_results['labels_clean'],
             valid_mask=label_results['valid_mask'],
             conditions=trial_info['conditions'])
    print(f"Binary labels saved to: {labels_file}")
    
    # Create simple bar chart
    import matplotlib.pyplot as plt
    
    conditions = trial_info['conditions']
    labels = label_results['labels']
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = {'Single': 'blue', 'SICI': 'red', 'SICF': 'green'}
    
    # Bar chart showing label distribution
    x_pos = 0
    for condition in ['Single', 'SICI', 'SICF']:
        mask = conditions == condition
        if np.any(mask):
            condition_labels = labels[mask]
            valid_labels = condition_labels[~np.isnan(condition_labels)]
            if len(valid_labels) > 0:
                counts = np.bincount(valid_labels.astype(int), minlength=2)
                
                # Plot bars - separate calls to avoid alpha list issue
                ax.bar(x_pos, counts[0], width=0.35, 
                      color=colors[condition], alpha=0.6,
                      label=f'{condition} Label 0' if x_pos == 0 else "")
                ax.bar(x_pos + 0.4, counts[1], width=0.35, 
                      color=colors[condition], alpha=1.0,
                      label=f'{condition} Label 1' if x_pos == 0 else "")
                
                # Add count labels
                for i, count in enumerate(counts):
                    ax.text(x_pos + i*0.4, count + 1, str(count), 
                           ha='center', va='bottom', fontweight='bold')
                
                x_pos += 1
    
    ax.set_xlabel('Condition')
    ax.set_ylabel('Number of Trials') 
    ax.set_title(f'{subject_id} - Binary Label Distribution (Median Split)')
    ax.set_xticks([0.2, 1.2, 2.2])
    ax.set_xticklabels(['Single\n(Low|High Ampl)', 'SICI\n(Worse|Better Inhib)', 'SICF\n(Worse|Better Facil)'])
    ax.legend(['Label 0', 'Label 1'], loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = output_path / f"{subject_id}_binary_labels_simple.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Label distribution plot saved to: {plot_filename}")

# Simple integration function
def add_binary_labels_to_main():
    # Create median split binary labels
    label_results = create_median_split_labels(trial_info)
    
    # Save labels and create simple visualization
    save_labels_simple(trial_info, label_results, SUBJECT_ID, FEATURES_OUTPUT_PATH)
    
    print(f"Binary labels created with balanced classes!")
    
    return label_results
    
def create_hjorth_filters(ch_names, center_electrodes):
    """
    Create Hjorth spatial filters for specified center electrodes.
    
    Args:
        ch_names (list): List of all channel names
        center_electrodes (list): List of center electrode names for Hjorth filters
        
    Returns:
        dict: Dictionary with filter matrices for each center electrode
    """
    # Define neighboring electrodes for each center electrode
    neighbor_map = {
        'F3': ['AF3', 'F1', 'FC3', 'F5'],
        'F4': ['AF4', 'F2', 'FC4', 'F6'], 
        'FC3': ['F3', 'FC1', 'C3', 'FC5'],
        'FC4': ['F4', 'FC2', 'C4', 'FC6'],
        'C3': ['FC3', 'C1', 'CP3', 'C5'],
        'C4': ['FC4', 'C2', 'CP4', 'C6'],
        'CP3': ['C3', 'CP1', 'P3', 'CP5'],
        'CP4': ['C4', 'CP2', 'P4', 'CP6'],
        'P3': ['CP3', 'P1', 'PO3', 'P5'],
        'P4': ['CP4', 'P2', 'PO4', 'P6'],
        'PO3': ['P3', 'PO7', 'O1', 'P5'],
        'PO4': ['P4', 'PO8', 'O2', 'P6']
    }
    
    hjorth_filters = {}
    n_channels = len(ch_names)
    
    for center_elec in center_electrodes:
        if center_elec not in ch_names:
            print(f"Warning: Center electrode {center_elec} not found in channel names")
            continue
            
        # Create filter vector
        filter_vec = np.zeros(n_channels)
        center_idx = ch_names.index(center_elec)
        filter_vec[center_idx] = 1.0  # Weight of 1 for center electrode
        
        # Find available neighbors and assign weights
        neighbors = neighbor_map.get(center_elec, [])
        available_neighbors = []
        
        for neighbor in neighbors:
            if neighbor in ch_names:
                neighbor_idx = ch_names.index(neighbor)
                available_neighbors.append(neighbor_idx)
        
        # Assign weight of -1/n_neighbors to each available neighbor
        if available_neighbors:
            neighbor_weight = -1.0 / len(available_neighbors)
            for neighbor_idx in available_neighbors:
                filter_vec[neighbor_idx] = neighbor_weight
            
            hjorth_filters[f'{center_elec}_hjorth'] = filter_vec
            print(f"Created Hjorth filter for {center_elec} with {len(available_neighbors)} neighbors")
        else:
            print(f"Warning: No neighbors found for {center_elec}")
    
    return hjorth_filters

def apply_hjorth_filters(data, hjorth_filters):
    """
    Apply Hjorth-style laplacian spatial filters to EEG data.
    """
    n_epochs, n_channels, n_times = data.shape
    n_hjorth = len(hjorth_filters)
    
    # Create filter matrix
    filter_names = list(hjorth_filters.keys())
    filter_matrix = np.array([hjorth_filters[name] for name in filter_names])
    
    # Apply filters: (n_hjorth, n_channels) @ (n_epochs, n_channels, n_times)
    filtered_data = np.einsum('hc,ect->eht', filter_matrix, data)
    
    return filtered_data, filter_names

def get_individual_alpha_peak(epochs, fmin=7, fmax=14, bw_scale=1.5):
    """
    Estimate individual alpha peak from EEG epochs.
    """
    print("Estimating individual alpha peak...")
    
    # Get data and sampling frequency
    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    sfreq = epochs.info['sfreq']
    
    # Calculate PSD across all epochs and channels
    n_times = data.shape[-1]
    bw = bw_scale * (sfreq / n_times)
    
    # Average across epochs and channels for alpha peak detection
    data_avg = np.mean(data, axis=(0, 1))  # Average across epochs and channels
    
    # Calculate PSD using multitaper method (similar to your approach)
    try:
        psds, freqs = mne.time_frequency.psd_array_multitaper(
            data_avg[np.newaxis, :], sfreq, 
            fmin=fmin, fmax=fmax, 
            bandwidth=bw, 
            output='power'
        )
        
        # Find peak in alpha range
        alpha_mask = (freqs >= fmin) & (freqs <= fmax)
        alpha_freqs = freqs[alpha_mask]
        alpha_psds = psds[0, alpha_mask]  # Remove singleton dimension
        
        # Find the frequency with maximum power
        peak_idx = np.argmax(alpha_psds)
        alpha_peak = alpha_freqs[peak_idx]
        
        print(f"Individual alpha peak found at: {alpha_peak:.2f} Hz")
        return alpha_peak, freqs
        
    except Exception as e:
        print(f"Error in alpha peak estimation: {e}")
        print("Using default alpha peak of 10.5 Hz")
        return False, None

def get_freq_ranges_based_on_alpha_peak(alpha_peak, freq_range_names):
    """
    Define frequency ranges based on individual alpha peak.
    """
    # Define frequency ranges relative to alpha peak
    theta_range = (alpha_peak - 6.5, alpha_peak - 1.5)  # ~4-8 Hz for 10.5 Hz alpha
    alpha_range = (alpha_peak - 2.5, alpha_peak + 2.5)  # ~8-13 Hz for 10.5 Hz alpha  
    beta_range = (alpha_peak + 2.5, alpha_peak + 14.5)  # ~13-25 Hz for 10.5 Hz alpha
    gamma_range = (alpha_peak + 14.5, alpha_peak + 36.5)  # ~25-47 Hz for 10.5 Hz alpha
    
    # But constrain to your data limits (2-47 Hz)
    theta_range = (max(2, theta_range[0]), min(47, theta_range[1]))
    alpha_range = (max(2, alpha_range[0]), min(47, alpha_range[1]))
    beta_range = (max(2, beta_range[0]), min(47, beta_range[1]))
    gamma_range = (max(2, gamma_range[0]), min(47, gamma_range[1]))
    
    freq_ranges = {
        'theta': theta_range,
        'alpha': alpha_range,
        'beta': beta_range,
        'gamma': gamma_range
    }
    
    return freq_ranges, freq_ranges

# %%
def get_bandpowers(data, fmin, fmax, sfreq, bw_scale):
    """
    Calculate bandpower using multitaper method.
    """
    n_times = data.shape[-1]
    bw = bw_scale * (sfreq / n_times)
    
    try:
        psds, freqs = mne.time_frequency.psd_array_multitaper(
            data, sfreq, 
            fmin=fmin, fmax=fmax, 
            bandwidth=bw, 
            output='power'
        )
        return psds, freqs, bw
    except Exception as e:
        print(f"Error calculating bandpowers for {fmin}-{fmax} Hz: {e}")
        return None, None, None

def extract_psd_features(epochs, freq_ranges, bw_scales, subject_id, output_path):
    """
    Extract PSD features for each frequency band and save them with channel names.
    """
    print("Extracting PSD features...")
    
    # Get data and sampling frequency
    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    sfreq = epochs.info['sfreq']
    n_epochs, n_channels, n_times = data.shape
    
    # Dictionary to store all features with interpretable names
    features_dict = {}
    
    # Extract features for each frequency band
    for freq_name, (fmin, fmax) in freq_ranges.items():
        print(f"Processing {freq_name} band: {fmin:.1f}-{fmax:.1f} Hz")
        
        # Get bandpowers for this frequency range
        psds, freqs, bw = get_bandpowers(data, fmin, fmax, sfreq, bw_scales[freq_name])
        
        if psds is not None:
            # Average across frequency bins to get bandpower per channel per epoch
            bandpowers = np.mean(psds, axis=2)  # Shape: (n_epochs, n_channels)
            
            # CHANGE: Save individual channel features with channel names
            for ch_idx, ch_name in enumerate(epochs.ch_names):
                feature_name = f"psd_{freq_name}_{ch_name}"
                features_dict[feature_name] = bandpowers[:, ch_idx]
            
            print(f"  {freq_name} bandpowers: {n_channels} channel features saved")
        else:
            print(f"  Failed to calculate {freq_name} bandpowers")
    
    # Save features
    features_file = output_path / f"{subject_id}_psd_features.npz"
    np.savez(features_file, **features_dict)
    print(f"Features saved to: {features_file}")
    
    # Also save metadata (unchanged)
    metadata = {
        'subject_id': subject_id,
        'freq_ranges': freq_ranges,
        'bw_scales': bw_scales,
        'n_epochs': n_epochs,
        'n_channels': n_channels,
        'channel_names': epochs.ch_names,
        'sfreq': sfreq
    }
    
    metadata_file = output_path / f"{subject_id}_metadata.npz"
    np.savez(metadata_file, **metadata)
    print(f"Metadata saved to: {metadata_file}")
    
    return features_dict, metadata

def extract_hjorth_wpli_features(epochs, freq_ranges, hjorth_filters, subject_id, output_path):
    """
    Extract WPLI connectivity features with connection pair names.
    """
    print("Extracting Hjorth WPLI features...")
    
    # Get data and sampling frequency
    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    sfreq = epochs.info['sfreq']
    
    # Apply Hjorth filters
    hjorth_data, hjorth_names = apply_hjorth_filters(data, hjorth_filters)
    print(f"Applied Hjorth filters. Filtered data shape: {hjorth_data.shape}")
    print(f"Hjorth channels: {hjorth_names}")
    
    # Dictionary to store WPLI features with connection names
    wpli_features_dict = {}
    
    # WPLI parameters
    use_wpli_params = {
        'theta': 2, 'alpha': 4, 'beta': 5, 'gamma': 6
    }

    # Extract WPLI for each frequency band
    for freq_name, (fmin, fmax) in freq_ranges.items():
        print(f"Processing WPLI for {freq_name} band: {fmin:.1f}-{fmax:.1f} Hz")
        
        # Get frequency array for this band
        n_times = hjorth_data.shape[-1]
        bw_scale = {'theta': 1.5, 'alpha': 1.5, 'beta': 2, 'gamma': 4}[freq_name]
        bw = bw_scale * (sfreq / n_times)
        freqs = np.arange(fmin + bw/2, fmax, bw)
        
        if len(freqs) == 0:
            print(f"  Warning: No frequencies in range for {freq_name}")
            continue
            
        # Calculate WPLI connectivity
        try:
            n_cycles = use_wpli_params[freq_name]
            
            # Use MNE connectivity function
            from mne_connectivity import spectral_connectivity_time
            
            con = spectral_connectivity_time(
                hjorth_data, 
                freqs=freqs, 
                method='wpli', 
                average=False,  # Keep trial-wise connectivity
                mode='multitaper',
                sfreq=sfreq, 
                faverage=True,  # Average over frequencies
                n_cycles=n_cycles,
                verbose=False,
                n_jobs=1
            )
            
            # Extract connectivity matrix
            wpli_data = con.get_data()  # Shape: (n_connections, n_epochs)
            wpli_data = wpli_data.T  # Shape: (n_epochs, n_connections)
            
            # CHANGE: Save individual connection features with pair names
            connection_idx = 0
            for i in range(len(hjorth_names)):
                for j in range(len(hjorth_names)):
                    if connection_idx < wpli_data.shape[1]:
                        ch1 = hjorth_names[i]
                        ch2 = hjorth_names[j]
                        feature_name = f"wpli_{freq_name}_{ch1}_{ch2}"
                        wpli_features_dict[feature_name] = wpli_data[:, connection_idx]
                        connection_idx += 1
            
            print(f"  {freq_name} WPLI: {wpli_data.shape[1]} connection features saved")
            
        except Exception as e:
            print(f"  Error calculating WPLI for {freq_name}: {e}")
    
    # Save WPLI features
    if wpli_features_dict:
        wpli_features_file = output_path / f"{subject_id}_hjorth_wpli_features.npz"
        np.savez(wpli_features_file, **wpli_features_dict)
        print(f"WPLI features saved to: {wpli_features_file}")
        
        # Save WPLI metadata
        wpli_metadata = {
            'subject_id': subject_id,
            'hjorth_channel_names': hjorth_names,
            'freq_ranges': freq_ranges,
            'n_hjorth_channels': len(hjorth_names),
            'wpli_params': use_wpli_params
        }
        
        wpli_metadata_file = output_path / f"{subject_id}_hjorth_wpli_metadata.npz"
        np.savez(wpli_metadata_file, **wpli_metadata)
        print(f"WPLI metadata saved to: {wpli_metadata_file}")
    
    return wpli_features_dict

def extract_hjorth_pac_features(epochs, freq_ranges, hjorth_filters, subject_id, output_path):
    """
    Extract PAC features with channel and frequency bin information.
    """
    print("Extracting Hjorth PAC features...")
    
    # Get data and sampling frequency
    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    sfreq = epochs.info['sfreq']
    
    # Apply Hjorth filters
    hjorth_data, hjorth_names = apply_hjorth_filters(data, hjorth_filters)
    print(f"Applied Hjorth filters for PAC. Filtered data shape: {hjorth_data.shape}")
    
    # Dictionary to store PAC features with interpretable names
    pac_features_dict = {}
    
    # Get frequency arrays for each band 
    freqs_arrays = {}
    for freq_identifier, (fmin, fmax) in freq_ranges.items():
        bw_scale = {'theta': 1.5, 'alpha': 1.5, 'beta': 2, 'gamma': 4}[freq_identifier]
        n_times = hjorth_data.shape[-1]
        bw = bw_scale * (sfreq / n_times)
        f_bins = np.arange(fmin + bw/2, fmax, bw)
        freqs_arrays[freq_identifier] = f_bins
    
    # Calculate PAC for all frequency band combinations
    freq_band_names = list(freq_ranges.keys())
    for freq_ind1, freq_identifier1 in enumerate(freq_band_names):
        for freq_ind2, freq_identifier2 in enumerate(freq_band_names):
            if freq_ind1 < freq_ind2:  # Only calculate upper triangular combinations
                print(f"  Processing PAC: {freq_identifier1}-{freq_identifier2}")
                
                freqs_lower = freqs_arrays[freq_identifier1]
                freqs_upper = freqs_arrays[freq_identifier2]
                
                try:
                    pac_data = get_hjorth_pac(hjorth_data, sfreq, freqs_lower, freqs_upper)
                    # pac_data shape: (n_epochs, n_hjorth_channels, n_freq_lower, n_freq_upper)
                    
                    # CHANGE: Save individual PAC features with channel and frequency info
                    n_epochs, n_channels, n_freq_low, n_freq_high = pac_data.shape
                    
                    for ch_idx, ch_name in enumerate(hjorth_names):
                        for freq_low_idx in range(n_freq_low):
                            for freq_high_idx in range(n_freq_high):
                                feature_name = f"pac_{freq_identifier1}_{freq_identifier2}_{ch_name}_f{freq_low_idx}_f{freq_high_idx}"
                                pac_features_dict[feature_name] = pac_data[:, ch_idx, freq_low_idx, freq_high_idx]
                    
                    print(f"    {freq_identifier1}-{freq_identifier2} PAC: {n_channels * n_freq_low * n_freq_high} features saved")
                    
                except Exception as e:
                    print(f"    Error calculating PAC for {freq_identifier1}-{freq_identifier2}: {e}")
    
    # Save PAC features
    if pac_features_dict:
        pac_features_file = output_path / f"{subject_id}_hjorth_pac_features.npz"
        np.savez(pac_features_file, **pac_features_dict)
        print(f"PAC features saved to: {pac_features_file}")
        
        # Save PAC metadata
        pac_metadata = {
            'subject_id': subject_id,
            'hjorth_channel_names': hjorth_names,
            'freq_ranges': freq_ranges,
            'freq_combinations': list(set([name.split('_')[1] + '_' + name.split('_')[2] for name in pac_features_dict.keys()])),
            'n_hjorth_channels': len(hjorth_names)
        }
        
        pac_metadata_file = output_path / f"{subject_id}_hjorth_pac_metadata.npz"
        np.savez(pac_metadata_file, **pac_metadata)
        print(f"PAC metadata saved to: {pac_metadata_file}")
    
    return pac_features_dict

def get_hjorth_pac(data, sfreq, freqs_lower, freqs_upper):
    """
    Calculate PAC for Hjorth-filtered data.
    """
    comods = []
    n_trials, n_hjorth_channels, n_times = data.shape
    
    for trial_ind in range(n_trials):
        data_trial = data[trial_ind, :, :]  # Data at this trial
        chans_pacs_trials = []
        
        for ch_ind in range(n_hjorth_channels):  # Go over all Hjorth channels
            ch_data_trial = data_trial[ch_ind, :]  # Data of this channel in this trial
            
            # Create PAC estimator
            estimator = pactools.Comodulogram(
                fs=sfreq, 
                low_fq_range=freqs_lower, 
                high_fq_range=freqs_upper, 
                method='tort',
                progress_bar=False, 
                random_state=42, 
                n_jobs=1
            )
            
            # Fit data and extract comodulogram
            estimator.fit(ch_data_trial)
            chans_pacs_trials.append(estimator.comod_)
        
        comods.append(np.array(chans_pacs_trials))
    
    return np.array(comods)

def design_bandpass_filter(low_freq, high_freq, sfreq, order=4):
    """
    Design a Butterworth bandpass filter for a specific frequency range.
    """
    nyquist = sfreq / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Ensure frequencies are within valid range
    low = max(low, 0.01)  # Avoid zero frequency
    high = min(high, 0.99)  # Avoid Nyquist frequency
    
    b, a = butter(order, [low, high], btype='band')
    return b, a

def phastimate_phase_estimation(data, filter_b, filter_a, edge, ar_order, hilbert_window, 
                               offset_correction=0, iterations=None, armethod='aryule'):
    """
    Estimate the phase of the EEG signal using autoregressive modeling and Hilbert transform.
    Adapted from the phastimate function (Zrenner et al., 2018)
    """
    if iterations is None:
        iterations = edge + int(np.ceil(hilbert_window / 2))

    # Ensure data length is sufficient for filtering
    padlen = 3 * (max(len(filter_a), len(filter_b)) - 1)
    if data.shape[0] <= padlen:
        return None, None  # Not enough data

    # Apply band-pass filter (BPF)
    data_filtered = filtfilt(filter_b, filter_a, data)

    # Remove edge samples to mitigate filter transients
    if data_filtered.shape[0] <= 2 * edge:
        return None, None  # Not enough data after removing edge samples
    data_no_edge = data_filtered[edge:-edge]

    # Determine AR parameters
    x = data_no_edge
    if len(x) < ar_order:
        return None, None  # Not enough data for AR model

    if armethod == 'aryule':
        a, _, _ = aryule(x, ar_order)
        actual_ar_order = len(a)  # Actual AR order used
        coefficients = -1 * a[::-1]  # Flip and negate all coefficients
    else:
        raise ValueError('Unknown AR method')

    # Prepare vector for forward prediction
    total_samples = len(data_no_edge) + iterations
    data_predicted = np.zeros(total_samples)
    data_predicted[:len(data_no_edge)] = data_no_edge

    # Extend the data array for forward prediction
    data_predicted = np.concatenate((data_no_edge, np.ones(iterations, dtype=np.float64)))

    # Run the forward prediction
    for i in range(iterations):
        idx = len(data_no_edge) + i
        data_segment = data_predicted[idx - actual_ar_order:idx]
        data_predicted[idx] = np.sum(coefficients * data_segment)

    # Extract the last hilbert_window samples for the Hilbert transform
    if data_predicted.shape[0] < hilbert_window:
        return None, None  # Not enough data for Hilbert transform
    hilbert_window_data = data_predicted[-hilbert_window:]

    # Compute the analytic signal and phase
    analytic_signal = hilbert(hilbert_window_data)
    phase = np.angle(analytic_signal)
    amplitude = np.abs(analytic_signal)

    return phase, amplitude

def extract_hjorth_phase_features(epochs, freq_ranges, hjorth_filters, subject_id, output_path):
    """
    Extract phase features with channel names.
    """
    print("Extracting Hjorth phase features...")
    
    # Get data and sampling frequency
    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    sfreq = epochs.info['sfreq']
    
    # Apply Hjorth filters
    hjorth_data, hjorth_names = apply_hjorth_filters(data, hjorth_filters)
    print(f"Applied Hjorth filters for phase. Filtered data shape: {hjorth_data.shape}")
    
    # Target Hjorth channels for phase estimation
    target_channels = ['FC3_hjorth', 'FC4_hjorth', 'C3_hjorth', 'C4_hjorth', 'CP3_hjorth', 'CP4_hjorth']
    
    # Find indices of target channels
    target_indices = []
    target_names = []
    for ch in target_channels:
        if ch in hjorth_names:
            target_indices.append(hjorth_names.index(ch))
            target_names.append(ch)
        else:
            print(f"Warning: {ch} not found in Hjorth channels")
    
    if not target_indices:
        print("Error: No target channels found for phase estimation")
        return {}
    
    print(f"Using channels for phase estimation: {target_names}")
    
    # Dictionary to store phase features with channel names
    phase_features_dict = {}
    
    # Phase estimation parameters
    edge = 50
    ar_order = 15
    hilbert_window = 80
    
    # Process each frequency band
    for freq_name, (fmin, fmax) in freq_ranges.items():
        print(f"Processing phase for {freq_name} band: {fmin:.1f}-{fmax:.1f} Hz")
        
        # Design bandpass filter for this frequency band
        try:
            filter_b, filter_a = design_bandpass_filter(fmin, fmax, sfreq, order=4)
        except Exception as e:
            print(f"  Error designing filter for {freq_name}: {e}")
            continue
        
        # Arrays to store sin/cos components for all epochs and target channels
        sin_components = []
        cos_components = []
        
        n_epochs = hjorth_data.shape[0]
        
        for epoch_idx in range(n_epochs):
            epoch_sin = []
            epoch_cos = []
            
            for ch_idx in target_indices:
                # Extract data for this channel and epoch
                ch_data = hjorth_data[epoch_idx, ch_idx, :]
                
                # Demean the data
                ch_data = ch_data - np.mean(ch_data)
                
                try:
                    # Estimate phase using phastimate
                    estimated_phases, estimated_amplitudes = phastimate_phase_estimation(
                        ch_data, filter_b, filter_a, edge, ar_order, hilbert_window
                    )
                    
                    if estimated_phases is not None:
                        final_phase = estimated_phases[-1]
                        sin_comp = np.sin(final_phase)
                        cos_comp = np.cos(final_phase)
                        epoch_sin.append(sin_comp)
                        epoch_cos.append(cos_comp)
                    else:
                        epoch_sin.append(np.nan)
                        epoch_cos.append(np.nan)
                        
                except Exception as e:
                    print(f"    Error in phase estimation for epoch {epoch_idx}, channel {ch_idx}: {e}")
                    epoch_sin.append(np.nan)
                    epoch_cos.append(np.nan)
            
            sin_components.append(epoch_sin)
            cos_components.append(epoch_cos)
        
        # Convert to numpy arrays
        sin_components = np.array(sin_components)  # Shape: (n_epochs, n_target_channels)
        cos_components = np.array(cos_components)  # Shape: (n_epochs, n_target_channels)
        
        # CHANGE: Save individual channel features with channel names
        for ch_idx, ch_name in enumerate(target_names):
            sin_feature_name = f"phase_{freq_name}_sin_{ch_name}"
            cos_feature_name = f"phase_{freq_name}_cos_{ch_name}"
            
            phase_features_dict[sin_feature_name] = sin_components[:, ch_idx]
            phase_features_dict[cos_feature_name] = cos_components[:, ch_idx]
        
        # Check for NaN values
        sin_nan_count = np.sum(np.isnan(sin_components))
        cos_nan_count = np.sum(np.isnan(cos_components))
        
        print(f"  {freq_name} phase: {len(target_names) * 2} features saved (NaN: sin={sin_nan_count}, cos={cos_nan_count})")
    
    # Save phase features
    if phase_features_dict:
        phase_features_file = output_path / f"{subject_id}_hjorth_phase_features.npz"
        np.savez(phase_features_file, **phase_features_dict)
        print(f"Phase features saved to: {phase_features_file}")
        
        # Save phase metadata
        phase_metadata = {
            'subject_id': subject_id,
            'target_channels': target_names,
            'target_channel_indices': target_indices,
            'freq_ranges': freq_ranges,
            'phase_params': {
                'edge': edge,
                'ar_order': ar_order,
                'hilbert_window': hilbert_window
            },
            'n_target_channels': len(target_indices)
        }
        
        phase_metadata_file = output_path / f"{subject_id}_hjorth_phase_metadata.npz"
        np.savez(phase_metadata_file, **phase_metadata)
        print(f"Phase metadata saved to: {phase_metadata_file}")
    
    return phase_features_dict

def apply_matplotlib_settings():
    """Apply consistent matplotlib settings."""
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

def create_mep_scatterplot(trial_info: dict, subject_id: str, output_path: Path):
    """
    Creates and saves a scatterplot of log-transformed MEP amplitudes, 
    colored by condition.
    """
    print("Creating MEP scatterplot...")

    # Apply the consistent styling
    apply_matplotlib_settings()

    # Extract necessary data from trial_info
    conditions = trial_info['conditions']
    # Use the log-transformed MEPs for the y-axis
    meps_log_transformed = trial_info['mep_log_transformed'] 

    # Define colors and markers to match feature_importance.py style
    colors = {
        'Single': '#cee1d1',  # Light green
        'SICI': '#80a687',    # Medium green
        'SICF': '#9e9e9e'     # Grey
    }
    markers = {'Single': 'o', 'SICI': 's', 'SICF': '^'}

    # Create the figure with the specified size (90mm x 45mm)
    fig_width_mm = 90
    fig_height_mm = 45
    fig, ax = plt.subplots(figsize=(fig_width_mm / 25.4, fig_height_mm / 25.4))

    # Plot data for each condition
    for condition in ['Single', 'SICI', 'SICF']:
        mask = (conditions == condition)
        if np.any(mask):
            # Use trial indices for the x-axis
            trial_indices = np.where(mask)[0]
            ax.scatter(
                trial_indices,
                meps_log_transformed[mask],
                c=colors[condition],
                marker=markers[condition],
                label=condition,
                s=8,  # Marker size
                alpha=1,
                edgecolor='none',
                linewidth=0.4
            )

    # --- Style the plot ---
    ax.set_xlabel('Trial number')
    ax.set_ylabel('Log MEP amplitude')
    
    # Add a legend below the plot in one line
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.3), # Position below the plot
        ncol=3                      # Arrange in a single row
    )

    
    # Ensure layout is tight
    plt.tight_layout()

    # --- Save the figure ---
    plot_filename = output_path / f"{subject_id}_mep_log_scatterplot.png"
    try:
        plt.savefig(plot_filename)
        print(f"MEP scatterplot saved to: {plot_filename}")
    except Exception as e:
        print(f"Error saving MEP scatterplot: {e}")
    finally:
        plt.close(fig) # Close the figure to free up memory
# %%
def main():
    """Main function that can be called with command line arguments."""
    #parser = argparse.ArgumentParser(description="Extract features for a single subject.")
    #parser.add_argument("--subject", required=True, type=str, help="The subject identifier (e.g., 'SICI-SICF_sub-01').")
    
    #args = parser.parse_args()
    SUBJECT_ID = "SICI-SICF_sub-07"

    #print(f"Starting feature extraction for subject: {args.subject}")
    print(f"Starting feature extraction for subject: {SUBJECT_ID}")
    
    # Get configuration
    #PROCESSED_DATA_PATH, FEATURES_OUTPUT_PATH, SUBJECT_ID = get_config_for_subject(args.subject)
    PROCESSED_DATA_PATH, FEATURES_OUTPUT_PATH, SUBJECT_ID = get_config_for_subject(SUBJECT_ID)
    
    # Load the data
    print("=== Loading Data ===")
    epochs, meps = load_eeg_and_mep_data(SUBJECT_ID, PROCESSED_DATA_PATH)

    if epochs is None or meps is None:
        print("Failed to load data. Please check your paths and try again.")
        sys.exit(1)
    else:
        print(f"Successfully loaded data:")
        print(f"  EEG epochs shape: {epochs.get_data().shape}")
        print(f"  MEPs shape: {meps.shape}")
        print(f"  Sampling frequency: {epochs.info['sfreq']} Hz")
        print(f"  Channels: {epochs.ch_names}")

    # Extract trial information, calculate ratios, and create visualization
    trial_info = extract_trial_info_and_ratios(epochs, meps, SUBJECT_ID, FEATURES_OUTPUT_PATH)

    # Create MEP scatterplot
    create_mep_scatterplot(trial_info, SUBJECT_ID, FEATURES_OUTPUT_PATH)
    
    # # Save individual condition arrays for easy access
    # conditions_file = FEATURES_OUTPUT_PATH / f"{SUBJECT_ID}_conditions.npy"
    # np.save(conditions_file, trial_info['conditions'])
    # print(f"Conditions saved to: {conditions_file}")

    # label_results = create_median_split_labels(trial_info)
    # save_labels_simple(trial_info, label_results, SUBJECT_ID, FEATURES_OUTPUT_PATH)
    
    # # Estimate individual alpha peak
    # print("\n=== Estimating Individual Alpha Peak ===")
    # alpha_peak, freqs = get_individual_alpha_peak(epochs, fmin=7, fmax=14, bw_scale=1.5)
    
    # # Use default if estimation failed
    # alpha_default = 10.5
    # alpha_peak_now = alpha_peak if alpha_peak is not False else alpha_default
    # is_default = alpha_peak is False
    
    # print(f"Using alpha peak: {alpha_peak_now:.2f} Hz (default: {is_default})")

    # # Define frequency ranges based on alpha peak
    # print("\n=== Defining Frequency Ranges ===")
    # freq_range_names = ['theta', 'alpha', 'beta', 'gamma']
    # freq_ranges, _ = get_freq_ranges_based_on_alpha_peak(alpha_peak_now, freq_range_names)
    
    # # Bandwidth scales (from your original script)
    # bw_scales = {'theta': 2, 'alpha': 2, 'beta': 3, 'gamma': 6}
    
    # print("Frequency ranges:")
    # for name, (fmin, fmax) in freq_ranges.items():
    #     print(f"  {name}: {fmin:.1f}-{fmax:.1f} Hz (bw_scale: {bw_scales[name]})")

    # # Create Hjorth spatial filters
    # print("\n=== Creating Hjorth Spatial Filters ===")
    # center_electrodes = ['F3', 'F4', 'FC3', 'FC4', 'C3', 'C4', 'CP3', 'CP4', 'P3', 'P4', 'PO3', 'PO4']
    # hjorth_filters = create_hjorth_filters(epochs.ch_names, center_electrodes)
    
    # print(f"Created {len(hjorth_filters)} Hjorth filters:")
    # for filter_name in hjorth_filters.keys():
    #     print(f"  {filter_name}")

    # # Uncomment and run whichever feature extractions you want:
    
    # # Extract PSD features
    # print("\n=== Extracting PSD Features ===")
    # features_dict, metadata = extract_psd_features(
    #     epochs, freq_ranges, bw_scales, SUBJECT_ID, FEATURES_OUTPUT_PATH
    # )
    
    # print("\nPSD feature extraction completed!")
    # print("Available PSD features:")
    # for feature_name, feature_data in features_dict.items():
    #     print(f"  {feature_name}: {feature_data.shape}")

    # # Extract Hjorth WPLI features
    # print("\n=== Extracting Hjorth WPLI Features ===")
    # wpli_features_dict = extract_hjorth_wpli_features(
    #     epochs, freq_ranges, hjorth_filters, SUBJECT_ID, FEATURES_OUTPUT_PATH
    # )
    
    # if wpli_features_dict:
    #     print("\nWPLI feature extraction completed!")
    #     print("Available WPLI features:")
    #     for feature_name, feature_data in wpli_features_dict.items():
    #         print(f"  {feature_name}: {feature_data.shape}")
    # else:
    #     print("No WPLI features extracted.")

    # # Extract Hjorth PAC features
    # print("\n=== Extracting Hjorth PAC Features ===")
    # pac_features_dict = extract_hjorth_pac_features(
    #     epochs, freq_ranges, hjorth_filters, SUBJECT_ID, FEATURES_OUTPUT_PATH
    # )
    
    # if pac_features_dict:
    #     print("\nPAC feature extraction completed!")
    #     print("Available PAC features:")
    #     for feature_name, feature_data in pac_features_dict.items():
    #         print(f"  {feature_name}: {feature_data.shape}")
    # else:
    #     print("No PAC features extracted.")

    # # Extract Hjorth Phase features
    # print("\n=== Extracting Hjorth Phase Features ===")
    # phase_features_dict = extract_hjorth_phase_features(
    #     epochs, freq_ranges, hjorth_filters, SUBJECT_ID, FEATURES_OUTPUT_PATH
    # )
    
    # if phase_features_dict:
    #     print("\nPhase feature extraction completed!")
    #     print("Available phase features:")
    #     for feature_name, feature_data in phase_features_dict.items():
    #         print(f"  {feature_name}: {feature_data.shape}")
    # else:
    #     print("No phase features extracted.")

    # # Save additional info
    # print("\n=== Saving Additional Info ===")
    
    # # Create frequency range dictionary with alpha peak info
    # freq_range_dict = freq_ranges.copy()
    # freq_range_dict['alpha_peak'] = alpha_peak_now
    # freq_range_dict['alpha_peak_is_default'] = is_default
    
    # # Save as .npy file 
    # freq_range_dict_path = FEATURES_OUTPUT_PATH / f'{SUBJECT_ID}_freq_ranges_dict.npy'
    # np.save(freq_range_dict_path, freq_range_dict)
    # print(f"Frequency ranges saved to: {freq_range_dict_path}")
    
    # # Also save MEPs for easy access later
    # meps_file = FEATURES_OUTPUT_PATH / f"{SUBJECT_ID}_MEPs.npy"
    # np.save(meps_file, meps)
    # print(f"MEPs saved to: {meps_file}")
    
    # print(f"\nAll files saved in: {FEATURES_OUTPUT_PATH}")
    # print(f"Feature extraction completed successfully for {SUBJECT_ID}!")

if __name__ == "__main__":
    main()
# %%
