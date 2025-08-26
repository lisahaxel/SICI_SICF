#%%
import pandas as pd
import mne
from pathlib import Path
from neurone_loader.loader import Session
import numpy as np
import matplotlib.pyplot as plt
import traceback

# --- Configuration ---
DATA_ROOT = Path('/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/SICISICF/Conventional')
LOG_FILE_ROOT = Path('/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/SICISICF/Conventional/mat_files_only/Conventional')
METADATA_FILE = Path('./pilot_sessions.xlsx')
NEURONE_INDEX_COLUMN = 'NeurOneIndex'
DATA_FOLDER_COLUMN = 'NeurOnePath'
LOG_FILE_COLUMN = 'log_file'
SAVE_ROOT = Path('./processed_data')
EPOCH_TMIN, EPOCH_TMAX = -1.005, 0.105


def process_block_and_epoch(subject_id, block_info, data_root, log_file_root, tmin, tmax):
    """
    Processes a single block by manually finding triggers based on their SourcePort and Type.
    """
    data_folder_name = block_info[DATA_FOLDER_COLUMN]
    log_file_name = block_info[LOG_FILE_COLUMN]
    neurone_phase_index = int(block_info[NEURONE_INDEX_COLUMN])
    full_block_path_id = f"{data_folder_name}/{neurone_phase_index}"
    print(f"  - Processing block: {full_block_path_id}")

    try:
        path_to_session_data = data_root / subject_id / str(data_folder_name)
        if not path_to_session_data.exists():
            print(f"      - ERROR: Session data path not found: {path_to_session_data}")
            return None

        session = Session(path_to_session_data)
        if not (0 < neurone_phase_index <= len(session.phases)):
            print(f"      - ERROR: Block index {neurone_phase_index} is invalid for session with {len(session.phases)} phases.")
            return None
        
        block = session.phases[neurone_phase_index - 1]
        
        log_filename_with_ext = f"{log_file_name}.csv" 
        log_file_path = log_file_root / subject_id / log_filename_with_ext
        if not log_file_path.exists():
            print(f"      - WARNING: Log file not found, skipping block: {log_file_path}")
            return None
            
        channel_mappings = {'TP9': 'eeg', 'TP10': 'eeg', 'APBr': 'emg', 'FDIr': 'emg', 'ADMr': 'emg'}
        mne_raw = block.to_mne(substitute_zero_events_with=999, channel_type_mappings=channel_mappings)
        mne_raw.rename_channels({'CPz': 'POz'})

        # print the channel names
        print(mne_raw.ch_names)
        
        labels_df = pd.read_csv(log_file_path, header=None)
        trial_types = labels_df[0].tolist()

        all_raw_events_df = block.events
        
        # --- FINAL FIX ---
        # Filter for triggers from Port 2 AND Type 1 (onset marker)
        port_b_triggers = all_raw_events_df[(all_raw_events_df['SourcePort'] == 2) & (all_raw_events_df['Type'] == 1)]

        if port_b_triggers.empty:
            print("      - ERROR: No triggers found from SourcePort 2 with Type 1.")
            return None
        
        print(f"      - Found {len(port_b_triggers)} triggers from Port 2, Type 1.")

        event_samples = port_b_triggers['StartSampleIndex'].values
        events = np.c_[event_samples, np.zeros(len(event_samples), dtype=int), np.ones(len(event_samples), dtype=int)]

        if len(events) != len(trial_types):
            print(f"      - WARNING: Mismatch between triggers ({len(events)}) and log trials ({len(trial_types)}). Truncating.")
            min_len = min(len(events), len(trial_types))
            events = events[:min_len]
            trial_types = trial_types[:min_len]

        event_id_map = {'Single': 1, 'SICI': 2, 'SICF': 3}
        new_events = np.copy(events)
        
        for i, trial_type in enumerate(trial_types):
            trial_type_clean = str(trial_type).strip()
            if trial_type_clean in event_id_map:
                new_events[i, 2] = event_id_map[trial_type_clean]
            else:
                new_events[i, 2] = 99

        epochs = mne.Epochs(
            mne_raw,
            events=new_events,
            event_id=event_id_map,
            tmin=tmin,
            tmax=tmax,
            proj=False,
            baseline=None,
            preload=True,
            reject=None
        )
        print(f"      - Successfully created {len(epochs)} epochs for block.")
        return epochs

    except Exception as e:
        print(f"      - An unhandled error occurred in block {full_block_path_id}: {e}")
        traceback.print_exc()
        return None

# --- Main execution ---
if __name__ == '__main__':
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        df_meta = pd.read_excel(METADATA_FILE)
    except FileNotFoundError:
        print(f"FATAL ERROR: Metadata file not found at {METADATA_FILE}")
        exit()
    if NEURONE_INDEX_COLUMN not in df_meta.columns:
        print(f"FATAL ERROR: The required column '{NEURONE_INDEX_COLUMN}' was not found in your Excel file.")
        exit()
            
    all_epochs_data = {}
    subjects_in_meta = df_meta['Subject'].unique()
    print(f"Found {len(subjects_in_meta)} unique subjects in the metadata file.")

    for subject_id, subject_blocks_df in df_meta.groupby('Subject'):
        print(f"\n--- Starting processing for subject: {subject_id} ---")
        
        subject_epochs_list = []
        subject_blocks_df = subject_blocks_df.sort_values('Block')

        for _, block_row in subject_blocks_df.iterrows():
            epochs_block = process_block_and_epoch(
                subject_id=subject_id,
                block_info=block_row,
                data_root=DATA_ROOT,
                log_file_root=LOG_FILE_ROOT,
                tmin=EPOCH_TMIN,
                tmax=EPOCH_TMAX
            )
            if epochs_block:
                subject_epochs_list.append(epochs_block)

        if not subject_epochs_list:
            print(f"--- No blocks could be processed for subject {subject_id}. Skipping. ---")
            continue

        print(f"\n- Concatenating {len(subject_epochs_list)} blocks for {subject_id}...")
        subject_epochs_all = mne.concatenate_epochs(subject_epochs_list)
        all_epochs_data[subject_id] = subject_epochs_all
        
        print(f"- Total epochs for {subject_id}: {len(subject_epochs_all)}")
        print(subject_epochs_all.event_id)

        save_path = SAVE_ROOT / f"{subject_id}-epo.fif"
        subject_epochs_all.save(save_path, overwrite=True, fmt='double')
        print(f"- Saved processed data to: {save_path}")
        print(f"--- Finished processing for subject: {subject_id} ---")

    print(f"\n--- Processing Complete ---")
    print(f"Successfully processed and saved data for {len(all_epochs_data)} out of {len(subjects_in_meta)} subjects.")


# %%
