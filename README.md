# SICI-SICF Classification Pipeline

The pipeline is structured as a series of scripts that must be run in order. Each script takes the output of the previous one as its input.

## Core Pipeline Overview

The workflow is divided into 6 main stages:

1.  **Data Loading:** Ingests raw multi-format data and converts it to MNE-Python epochs.
2.  **Preprocessing:** Cleans the epoch data by removing channel, trial, and component-level artifacts.
3.  **Feature Extraction:** Engineers a wide range of features (power, connectivity, phase) from the clean EEG data and creates classification labels from MEPs.
4.  **Model Training:** Trains and evaluates machine learning models using Bayesian optimization (Optuna) and mRMR feature selection.
5.  **Results Visualization:** Aggregates performance from all models and subjects to generate summary plots.
6.  **Feature Importance:** (Script not provided, but is the logical next step) Analyzes which features were most predictive.

---

##  Script-by-Script Breakdown

Here is a description of each script in the order they should be run.

### 1. `load_data.py`

* **Purpose:** To load raw, multi-block data from the NeurOne system, combine it with experimental log files, and save it as standardized MNE `Epochs` files.
* **Input:**
    * Raw NeurOne session data (from `DATA_ROOT`).
    * `.csv` log files containing trial condition labels (from `LOG_FILE_ROOT`).
    * `pilot_sessions.xlsx` metadata file to map subjects to their respective data and log files.
* **Key Operations:**
    * Uses `neurone_loader` to read session data.
    * Parses log files to create event IDs (`Single`, `SICI`, `SICF`).
    * Manually aligns triggers with log file labels.
    * Concatenates all blocks for a single subject into one file.
* **Output:**
    * `processed_data/[subject_id]-epo.fif` (A single MNE Epochs file per subject).

### 2. `preprocessing_single_subject_offline.py`

* **Purpose:** To take a raw `epo.fif` file and perform offline artifact rejection. This is the main preprocessing step.
* **Input:**
    * `processed_data/[subject_id]-epo.fif`
* **Key Operations:**
    * **Channel Rejection:** Identifies and interpolates bad EEG channels based on median absolute deviation (MAD), high-frequency power, and autocorrelation.
    * **ICA Calibration:** Trains an Independent Component Analysis (ICA) model to identify and flag artifactual components (e.g., eye blinks).
    * **Trial Rejection (EEG):** Removes trials with excessive amplitudes (global and local z-scoring).
    * **Trial Rejection (EMG):** Selects the best EMG channel and rejects trials based on pre-innervation or invalid peak-to-peak (PTP) amplitude.
    * **Data Saving:** Saves the cleaned pre-stimulus EEG, the cleaned EMG epochs, and the MEP PTP values.
* **Output:**
    * `data_processed_final_.../[subject_id]/[subject_id]_pre-epo.fif` (Cleaned pre-stimulus EEG)
    * `data_processed_final_.../[subject_id]/[subject_id]_emg-epo.fif` (Cleaned EMG)
    * `data_processed_final_.../[subject_id]/[subject_id]_MEPs.npy` (PTP values for all *accepted* trials)
    * `data_processed_final_.../[subject_id]/[subject_id]_ica.fif` (The fitted ICA model)
    * `data_processed_final_.../[subject_id]/[subject_id]_preprocessing_info.npz` (A file containing all preprocessing parameters, rejected channels, etc.)

### 3. `get_features.py`

* **Purpose:** To convert the cleaned, time-domain EEG data into a static feature matrix for machine learning.
* **Input:**
    * `data_processed_final_.../[subject_id]/[subject_id]_pre-epo.fif`
    * `data_processed_final_.../[subject_id]/[subject_id]_MEPs.npy`
* **Key Operations:**
    * **Label Creation:** Loads MEPs and creates binary classification labels for each condition (`SICI`, `SICF`, `Single`) based on a median split.
    * **Spatial Filtering:** Applies Hjorth (Laplacian) filters to create "virtual" channels with higher spatial specificity.
    * **Feature Engineering:** Extracts several feature types from the pre-stimulus window:
        * Power Spectral Density (PSD)
        * Weighted Phase Lag Index (WPLI)
        * Phase-Amplitude Coupling (PAC)
        * Oscillatory Phase (e.g., `sin(phase)`, `cos(phase)`)
* **Output:**
    * `features_extracted/[subject_id]_binary_labels.npz` (The `y` values for the model)
    * `features_extracted/[subject_id]_psd_features.npz` (And others like `_wpli_`, `_pac_`, `_phase_`)
    * `features_extracted/[subject_id]_trial_info.npz` (MEP ratios, block info)
    * Various summary plots (`.png`) for data visualization.

### 4. `main_ml_optimized.py`

* **Purpose:** To run the complete machine learning analysis using the extracted features and labels.
* **Input:**
    * All feature files from `features_extracted/`.
* **Key Operations:**
    * **Data Loading:** Loads features and labels for all subjects.
    * **Feature Selection:** Uses `mRMR_feature_select` to find the most relevant features.
    * **Hyperparameter Tuning:** Uses `optuna` for Bayesian optimization of model (SVM, LogReg, RF) hyperparameters.
    * **Analyses:**
        1.  **Within-Subject:** Trains and evaluates models using cross-validation on a per-subject basis.
        2.  **Leave-One-Subject-Out (LOSO):** Performs transfer learning by training on all subjects *except one* and testing on the held-out subject.
* **Output:**
    * `results/models/LOSO_results_[condition].pkl` (Results for transfer learning)
    * `results/models/[subject_id]_[condition]_results.pkl` (Results for within-subject)

### 5. `performance_results.py`

* **Purpose:** To aggregate all model results and generate a final performance summary figure.
* **Input:**
    * All `.pkl` result files from `results/models/`.
* **Key Operations:**
    * Loads all within-subject and LOSO results.
    * Calculates ROC-AUC scores for every subject and model.
    * Generates a combined violin and box plot comparing:
        * Within-subject (SS)
        * LOSO Zero-shot (TL)
        * LOSO Calibrated (TL)
    * Prints a text summary of (Mean ± Std) performance to the console.
* **Output:**
    * `roc_auc_performance.pdf` (Final summary plot)
    * A performance summary printed to the terminal.

---

##  How to Run the Pipeline

The pipeline must be run sequentially.

1.  **Load Data:**
    ```bash
    python load_data.py
    ```

2.  **Preprocess Data:**
    This script must be run once for *each subject*. It takes the subject ID as a command-line argument.
    ```bash
    python preprocessing_single_subject_offline.py --subject SICI-SICF_sub-01
    python preprocessing_single_subject_offline.py --subject SICI-SICF_sub-02
    # ...and so on for all subjects
    ```

3.  **Extract Features:**
    This script also needs to be run for each subject.
    ```bash
    python get_features.py
    ```
    **Note:** As of this writing, `get_features.py` has the subject ID hardcoded in its `main()` function. You will need to edit this file or add `argparse` to loop through all subjects.

4.  **Run ML Analysis:**
    This script runs the analysis for all subjects and conditions.
    ```bash
    python main_ml_optimized.py
    ```

5.  **Generate Results Plot:**
    This script collects all results and creates the final figure.
    ```bash
    python performance_results.py
    ```

##  Key Dependencies

Make sure you have the following key libraries installed in your environment:

* `mne`
* `mne-connectivity`
* `neurone_loader` (for loading the raw data)
* `optuna` (for Bayesian hyperparameter optimization)
* `pactools` (for Phase-Amplitude Coupling)
* `spectrum` (for AR models, used in phase estimation)
* `pandas`
* `scipy`
* `scikit-learn`
* `matplotlib`
