# %%
import logging
import re
from pathlib import Path
from typing import List, Optional, Union

import mne
import numpy as np
import pandas as pd
from mne import BaseEpochs
from moabb.datasets.base import BaseDataset
from moabb.paradigms.base import BaseParadigm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from tqdm.auto import tqdm
from scipy.stats import ecdf
import matplotlib.pyplot as plt

# Setup logger for the entire module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

# Data paths for SICI/SICF analysis
SICI_SICF_DATA_ROOT_PATH = Path("/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/SICISICF/data_processed_final_pre_ica_True_final")
#SICI_SICF_DATA_ROOT_PATH = Path("/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/SICISICF/data_processed_final_pre_ica_True_final_offline")

# %%
# Real-Time Compatible Labeler with Warm-up 
class RealTimeLabeler:
    """
    A stateful labeler that learns from a calibration set and applies transformations
    causally to generate soft, probabilistic labels.
    """
    def __init__(
        self,
        target_col: str,
        scale_factor: float = 1.0,
        ewma_span: int = 25,
    ):
        self.target_col = target_col
        self.scale_factor = scale_factor
        self.ewma_span = ewma_span
        # --- Ignore the initial unstable period of EWMA when fitting ---
        self.warmup_period = ewma_span
        self.is_fitted = False
        self.cal_mean_ = 0
        self.cal_std_ = 1
        self.cdf_function_ = None

    def fit(self, metadata_df_cal: pd.DataFrame):
        """
        Learns normalization stats and the ECDF from the calibration block,
        ignoring the initial EWMA warm-up period for stability.
        """
        metadata_copy = metadata_df_cal.copy()
        values = metadata_copy[self.target_col].astype(float) * self.scale_factor

        ewma_trend = values.ewm(span=self.ewma_span, adjust=True).mean()
        detrended_values = values - ewma_trend

        # --- Use warmup period to learn from the stable part of the signal ---
        stable_detrended_values = detrended_values[self.warmup_period:]

        self.cal_mean_ = np.nanmean(stable_detrended_values)
        self.cal_std_ = np.nanstd(stable_detrended_values)
        if self.cal_std_ < 1e-9:
            self.cal_std_ = 1

        normalized_values = (stable_detrended_values - self.cal_mean_) / self.cal_std_
        self.cdf_function_ = ecdf(normalized_values.dropna()).cdf.evaluate
        self.is_fitted = True
        return self

    def transform(self, metadata_df_full: pd.DataFrame) -> np.ndarray:
        """
        Applies the learned transformations to the full session's data.
        """
        if not self.is_fitted:
            raise RuntimeError("The labeler has not been fitted yet. Call .fit() first.")

        metadata_copy = metadata_df_full.copy()
        values = metadata_copy[self.target_col].astype(float) * self.scale_factor
        
        ewma_trend = values.ewm(span=self.ewma_span, adjust=True).mean()
        detrended_values = values - ewma_trend
        
        normalized_values = (detrended_values - self.cal_mean_) / self.cal_std_
        soft_labels = normalized_values.apply(self.cdf_function_)
        
        return soft_labels.values


# %%
def calculate_sici_sicf_ratios(meps: np.ndarray, conditions: np.ndarray) -> np.ndarray:
    """
    Calculate SICI/SICF ratios based on block structure.
    """
    # Convert to microvolts and log-transform
    meps_uv = meps * 1e6
    mep_log_transformed = np.log1p(np.abs(meps_uv) + 1e-12)
    
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
    blocks = identify_blocks(conditions)
    
    # Initialize ratios array
    mep_ratios = np.full(len(meps), np.nan)
    
    # Calculate ratios for each block
    for block_num, trial_indices_in_block in enumerate(blocks):
        block_conditions = conditions[trial_indices_in_block]
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
    
    return mep_ratios


def extract_condition_labels(epochs: BaseEpochs) -> np.ndarray:
    """
    Extract condition labels from MNE epochs events.
    """
    # Create proper mapping: 1->Single, 2->SICI, 3->SICF
    event_code_mapping = {1: 'Single', 2: 'SICI', 3: 'SICF'}
    
    conditions = np.array([None] * len(epochs))
    
    for i, event_code in enumerate(epochs.events[:, 2]):
        if event_code in event_code_mapping:
            conditions[i] = event_code_mapping[event_code]
        else:
            log.warning(f"Unknown event code {event_code} at trial {i}")
            conditions[i] = f"Unknown_{event_code}"
    
    return conditions


# %%
class _BaseSICISICFDataset(BaseDataset):
    """Base dataset for SICI/SICF analysis with shared functionality."""
    
    def __init__(self, data_path: Union[str, Path, None] = None, subject_list: Union[List[int], None] = None):
        self.data_path_root = Path(data_path) if data_path else SICI_SICF_DATA_ROOT_PATH
        effective_subject_list = subject_list if subject_list is not None else self._discover_subjects()
        
        # Use integer subject IDs as required by MOABB
        super().__init__(
            subjects=effective_subject_list, 
            sessions_per_subject=1, 
            events={"TMS_stim": 1},
            code="SICISICFDataset", 
            interval=[-0.505, -0.006], 
            paradigm="sici_sicf_tms_eeg", 
            doi=None
        )

    def _discover_subjects(self) -> List[int]:
        """Discover available SICI-SICF subjects and return integer IDs."""
        subjects = []
        if not self.data_path_root.is_dir(): 
            return []
            
        for subject_dir in self.data_path_root.glob("SICI-SICF_sub-*"):
            if subject_dir.is_dir():
                # Extract the numeric part from SICI-SICF_sub-XX
                match = re.search(r"SICI-SICF_sub-(\d+)", subject_dir.name)
                if match:
                    subjects.append(int(match.group(1)))
        
        return sorted(subjects)

    def _subject_int_to_str(self, subject_int: int) -> str:
        """Convert integer subject ID to string format used in filenames."""
        return f"SICI-SICF_sub-{subject_int:02d}"

    def _get_single_subject_data(self, subject: int) -> dict:
        """Load and process data for a single subject."""
        subject_str = self._subject_int_to_str(subject)
        eeg_file = self.data_path_root / subject_str / f"{subject_str}_pre-epo.fif"
        mep_file = self.data_path_root / subject_str / f"{subject_str}_MEPs.npy"

        if not all([f.exists() for f in [eeg_file, mep_file]]):
            log.warning(f"Data files missing for {subject_str}. Skipping.")
            return {}

        try:
            # Load data
            epochs = mne.read_epochs(eeg_file, preload=True, verbose=False)
            meps = np.load(mep_file)
            
            # Fix MEPs shape if needed (from get_features.py logic)
            if meps.ndim == 2 and meps.shape[0] == 1:
                meps = meps.squeeze(0)
            
            # Extract condition labels from epochs
            conditions = extract_condition_labels(epochs)
            
            if not (len(epochs) == len(meps) == len(conditions)):
                log.error(f"{subject_str}: Mismatch in data lengths. Skipping.")
                return {}
            
            # Calculate ratios
            mep_ratios = calculate_sici_sicf_ratios(meps, conditions)
            
            # Create metadata with all relevant information
            metadata = pd.DataFrame({
                "MEP_value": meps,
                "MEP_ratio": mep_ratios,
                "condition": conditions,
                "MEP_log": np.log1p(np.abs(meps * 1e6) + 1e-12)  # For Single pulse analysis
            })
            
            epochs.metadata = metadata
            return {"0": {"0": epochs}}
            
        except Exception as e:
            log.error(f"{subject_str}: Error loading data: {e}")
            return {}

    def data_path(self, subject: int, **kwargs) -> List[str]:
        """Return data paths for a subject."""
        subject_str = self._subject_int_to_str(subject)
        paths = [
            self.data_path_root / subject_str / f"{subject_str}_pre-epo.fif",
            self.data_path_root / subject_str / f"{subject_str}_MEPs.npy"
        ]
        return [str(p) for p in paths if p.exists()]


class SICIDataset(_BaseSICISICFDataset):
    """Dataset specifically for SICI trials."""
    
    def __init__(self, data_path: Union[str, Path, None] = None, subject_list: Union[List[int], None] = None):
        super().__init__(data_path, subject_list)
        self.code = "SICIDataset"

    def _get_single_subject_data(self, subject: int) -> dict:
        """Load data and filter for SICI trials only."""
        data_dict = super()._get_single_subject_data(subject)
        
        if not data_dict:
            return {}
            
        epochs = data_dict["0"]["0"]
        
        # Filter for SICI trials only
        sici_mask = epochs.metadata['condition'] == 'SICI'
        if not np.any(sici_mask):
            subject_str = self._subject_int_to_str(subject)
            log.warning(f"{subject_str}: No SICI trials found. Skipping.")
            return {}
            
        epochs_sici = epochs[sici_mask]
        subject_str = self._subject_int_to_str(subject)
        log.info(f"{subject_str}: Found {len(epochs_sici)} SICI trials")
        
        return {"0": {"0": epochs_sici}}


class SICFDataset(_BaseSICISICFDataset):
    """Dataset specifically for SICF trials."""
    
    def __init__(self, data_path: Union[str, Path, None] = None, subject_list: Union[List[int], None] = None):
        super().__init__(data_path, subject_list)
        self.code = "SICFDataset"

    def _get_single_subject_data(self, subject: int) -> dict:
        """Load data and filter for SICF trials only."""
        data_dict = super()._get_single_subject_data(subject)
        
        if not data_dict:
            return {}
            
        epochs = data_dict["0"]["0"]
        
        # Filter for SICF trials only
        sicf_mask = epochs.metadata['condition'] == 'SICF'
        if not np.any(sicf_mask):
            subject_str = self._subject_int_to_str(subject)
            log.warning(f"{subject_str}: No SICF trials found. Skipping.")
            return {}
            
        epochs_sicf = epochs[sicf_mask]
        subject_str = self._subject_int_to_str(subject)
        log.info(f"{subject_str}: Found {len(epochs_sicf)} SICF trials")
        
        return {"0": {"0": epochs_sicf}}


class SinglePulseDataset(_BaseSICISICFDataset):
    """Dataset specifically for Single pulse trials."""
    
    def __init__(self, data_path: Union[str, Path, None] = None, subject_list: Union[List[int], None] = None):
        super().__init__(data_path, subject_list)
        self.code = "SinglePulseDataset"

    def _get_single_subject_data(self, subject: int) -> dict:
        """Load data and filter for Single pulse trials only."""
        data_dict = super()._get_single_subject_data(subject)
        
        if not data_dict:
            return {}
            
        epochs = data_dict["0"]["0"]
        
        # Filter for Single pulse trials only
        single_mask = epochs.metadata['condition'] == 'Single'
        if not np.any(single_mask):
            subject_str = self._subject_int_to_str(subject)
            log.warning(f"{subject_str}: No Single pulse trials found. Skipping.")
            return {}
            
        epochs_single = epochs[single_mask]
        subject_str = self._subject_int_to_str(subject)
        log.info(f"{subject_str}: Found {len(epochs_single)} Single pulse trials")
        
        return {"0": {"0": epochs_single}}


# %%
class _BaseSICISICFParadigm(BaseParadigm):
    """
    Base class for SICI/SICF paradigms, uses a fixed number of initial
    trials for calibration.
    """
    def __init__(
        self,
        tmin: float,
        tmax: float,
        target_metadata_col: str,
        dataset_class,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(filters=[], **kwargs)
        self.tmin = tmin
        self.tmax = tmax
        self.target_metadata_col = target_metadata_col
        self.dataset_class = dataset_class
        self.calibration_trials = 50  # Reduced for SICI/SICF as there are fewer trials per condition

    @property
    def datasets(self):
        return [self.dataset_class()]

    def is_valid(self, dataset):
        return "sici_sicf" in dataset.paradigm

    def make_labels_pipeline(self):
        raise NotImplementedError("Subclasses must implement their own label pipeline.")

    def get_data(self, dataset, subjects=None, return_epochs=False):
        """
        Main method to retrieve and process data.
        """
        if not self.is_valid(dataset):
            raise ValueError(f"Dataset {dataset.code} is not compatible.")

        subject_list = subjects if subjects is not None else dataset.subject_list
        raw_epochs_data = dataset.get_data(subject_list)

        X_list, y_list, metadata_list = [], [], []

        for subject in tqdm(subject_list, desc=f"Processing subjects for {self.__class__.__name__}"):
            if subject not in raw_epochs_data:
                continue

            epochs = raw_epochs_data[subject]["0"]["0"]
            epochs.crop(tmin=self.tmin, tmax=self.tmax, include_tmax=True)
            
            full_metadata = epochs.metadata.copy()

            if len(full_metadata) < self.calibration_trials:
                subject_str = f"SICI-SICF_sub-{subject:02d}"
                log.warning(
                    f"{subject_str}: Needs at least {self.calibration_trials} trials "
                    f"for calibration, but found only {len(full_metadata)}. Skipping."
                )
                continue

            meta_calibration = full_metadata.iloc[:self.calibration_trials]

            labeler = self.make_labels_pipeline()
            labeler.fit(meta_calibration)
            y_run = labeler.transform(full_metadata)

            nan_mask = np.isnan(y_run)
            if np.any(nan_mask):
                subject_str = f"SICI-SICF_sub-{subject:02d}"
                log.warning(f"{subject_str}: Found {np.sum(nan_mask)} NaN labels. Removing corresponding trials.")
                epochs = epochs[~nan_mask]
                y_run = y_run[~nan_mask]
                full_metadata = full_metadata[~nan_mask]

            if len(epochs) == 0:
                continue

            y_list.append(y_run)
            metadata_list.append(full_metadata)
            X_list.append(epochs.get_data(copy=False) if not return_epochs else epochs)

        if not X_list:
            return np.array([]), np.array([]), pd.DataFrame()

        metadata_final = pd.concat(metadata_list, ignore_index=True)
        y_final = np.concatenate(y_list)
        X_final = np.concatenate(X_list, axis=0) if not return_epochs else mne.concatenate_epochs(X_list)

        log.info(f"Final data shapes - X: {X_final.shape}, y: {y_final.shape}")
        return X_final, y_final, metadata_final

    def used_events(self, dataset):
        """Returns the event dictionary defined in this paradigm."""
        if hasattr(self, 'events'):
            return self.events
        return None


# %%
class SICIClassification(_BaseSICISICFParadigm):
    """
    Classification paradigm for SICI trials using MEP ratios.
    Lower ratios indicate better inhibition (higher class probability).
    """
    def __init__(self, tmin: float = -0.5, tmax: float = -0.020, **kwargs):
        super().__init__(
            tmin=tmin, 
            tmax=tmax, 
            target_metadata_col="MEP_ratio",
            dataset_class=SICIDataset,
            events={"TMS_stim": 1}, 
            **kwargs
        )

    @property
    def scoring(self):
        return "roc_auc"

    def make_labels_pipeline(self):
        # For SICI: lower ratios = better inhibition = higher probability
        # We'll invert the ratio so that better inhibition gets higher labels
        class SICILabeler(RealTimeLabeler):
            def transform(self, metadata_df_full: pd.DataFrame) -> np.ndarray:
                # Get the standard soft labels
                soft_labels = super().transform(metadata_df_full)
                # Invert for SICI: lower ratio = higher label (better inhibition)
                return 1.0 - soft_labels
        
        return SICILabeler(target_col=self.target_metadata_col, scale_factor=1.0)


class SICFClassification(_BaseSICISICFParadigm):
    """
    Classification paradigm for SICF trials using MEP ratios.
    Higher ratios indicate better facilitation (higher class probability).
    """
    def __init__(self, tmin: float = -0.5, tmax: float = -0.020, **kwargs):
        super().__init__(
            tmin=tmin, 
            tmax=tmax, 
            target_metadata_col="MEP_ratio",
            dataset_class=SICFDataset,
            events={"TMS_stim": 1}, 
            **kwargs
        )

    @property
    def scoring(self):
        return "roc_auc"

    def make_labels_pipeline(self):
        # For SICF: higher ratios = better facilitation = higher probability
        return RealTimeLabeler(target_col=self.target_metadata_col, scale_factor=1.0)


class SinglePulseClassification(_BaseSICISICFParadigm):
    """
    Classification paradigm for Single pulse trials using log-transformed MEP amplitudes.
    Higher amplitudes indicate higher excitability (higher class probability).
    """
    def __init__(self, tmin: float = -0.5, tmax: float = -0.020, **kwargs):
        super().__init__(
            tmin=tmin, 
            tmax=tmax, 
            target_metadata_col="MEP_log",
            dataset_class=SinglePulseDataset,
            events={"TMS_stim": 1}, 
            **kwargs
        )

    @property
    def scoring(self):
        return "roc_auc"

    def make_labels_pipeline(self):
        # For Single pulse: higher log MEP = higher excitability = higher probability
        return RealTimeLabeler(target_col=self.target_metadata_col, scale_factor=1.0)


# %%
def plot_sici_sicf_diagnostics(subject_id, metadata, final_labels, condition_type, target_col):
    """Generates diagnostic plots for SICI/SICF labeling process."""
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f"{condition_type} Real-Time Labeling Diagnostics for Subject {subject_id}", fontsize=16)

    raw_values = metadata[target_col].values
    trials = np.arange(len(raw_values))
    
    # Panel 1: Raw Values (Ratios or Log MEPs)
    axes[0].scatter(trials, raw_values, alpha=0.7, s=20, color='blue')
    axes[0].set_title(f"1. Raw {target_col} Values")
    axes[0].set_ylabel("Amplitude/Ratio")
    axes[0].grid(True, linestyle="--")

    # Panel 2: Final Soft Labels
    axes[1].scatter(trials, final_labels, alpha=0.7, s=20, color='red')
    axes[1].set_title("2. Final Probabilistic 'Soft' Labels")
    axes[1].set_xlabel("Trial Number")
    axes[1].set_ylabel("Soft Label [0-1]")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, linestyle="--")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    test_subject = 1  # This corresponds to SICI-SICF_sub-01
    
    # --- SICI ANALYSIS ---
    log.info(f"\n{'='*25} RUNNING SICI EXAMPLE (Subject {test_subject}) {'='*25}")
    dataset_sici = SICIDataset(subject_list=[test_subject])

    if test_subject in dataset_sici.subject_list:
        paradigm_sici = SICIClassification(tmin=-0.5, tmax=-0.020)
        X_sici, y_sici, meta_sici = paradigm_sici.get_data(dataset_sici)

        if y_sici.size > 0:
            log.info("Generating diagnostic plot for SICI processing...")
            plot_sici_sicf_diagnostics(
                subject_id=f"SICI-SICF_sub-{test_subject:02d}",
                metadata=meta_sici,
                final_labels=y_sici,
                condition_type="SICI",
                target_col='MEP_ratio'
            )
            log.info(f"SICI analysis complete: {len(y_sici)} trials, label range: {y_sici.min():.3f}-{y_sici.max():.3f}")
        else:
            log.warning("No SICI data available to plot.")
    else:
        log.error(f"Subject {test_subject} not found for SICI analysis.")

    # --- SICF ANALYSIS ---
    log.info(f"\n{'='*25} RUNNING SICF EXAMPLE (Subject {test_subject}) {'='*25}")
    dataset_sicf = SICFDataset(subject_list=[test_subject])

    if test_subject in dataset_sicf.subject_list:
        paradigm_sicf = SICFClassification(tmin=-0.5, tmax=-0.020)
        X_sicf, y_sicf, meta_sicf = paradigm_sicf.get_data(dataset_sicf)

        if y_sicf.size > 0:
            log.info("Generating diagnostic plot for SICF processing...")
            plot_sici_sicf_diagnostics(
                subject_id=f"SICI-SICF_sub-{test_subject:02d}",
                metadata=meta_sicf,
                final_labels=y_sicf,
                condition_type="SICF",
                target_col='MEP_ratio'
            )
            log.info(f"SICF analysis complete: {len(y_sicf)} trials, label range: {y_sicf.min():.3f}-{y_sicf.max():.3f}")
        else:
            log.warning("No SICF data available to plot.")
    else:
        log.error(f"Subject {test_subject} not found for SICF analysis.")

    # --- SINGLE PULSE ANALYSIS ---
    log.info(f"\n{'='*25} RUNNING SINGLE PULSE EXAMPLE (Subject {test_subject}) {'='*25}")
    dataset_single = SinglePulseDataset(subject_list=[test_subject])

    if test_subject in dataset_single.subject_list:
        paradigm_single = SinglePulseClassification(tmin=-0.5, tmax=-0.020)
        X_single, y_single, meta_single = paradigm_single.get_data(dataset_single)

        if y_single.size > 0:
            log.info("Generating diagnostic plot for Single Pulse processing...")
            plot_sici_sicf_diagnostics(
                subject_id=f"SICI-SICF_sub-{test_subject:02d}",
                metadata=meta_single,
                final_labels=y_single,
                condition_type="Single Pulse",
                target_col='MEP_log'
            )
            log.info(f"Single Pulse analysis complete: {len(y_single)} trials, label range: {y_single.min():.3f}-{y_single.max():.3f}")
        else:
            log.warning("No Single Pulse data available to plot.")
    else:
        log.error(f"Subject {test_subject} not found for Single Pulse analysis.")
# %%
