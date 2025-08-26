# %%
"""
SLURM Job Submission Script for EEG Transfer Learning Experiments
"""

import argparse
import itertools
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import submitit

# %%
# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Repository and environment setup
REPO_DIR = Path(f"/mnt/lustre/work/macke/{os.environ.get('USER', 'user')}/repos/eegjepa")
CONDA_ENV = "timeseries"
SCRIPT_TO_RUN = REPO_DIR / "EDAPT_neurips/EDAPT_TMS/SICISICF/PRIME_SICISICF/train_transfer.py"
BASE_OUTPUT_DIR = REPO_DIR / "EDAPT_neurips/EDAPT_TMS/SICISICF/PRIME_SICISICF/results_prime"

# SLURM cluster configuration
SLURM_PARTITION = "a100-galvani"  # Alternative: "2080"
MEM_GB_PER_GPU = 96
CPUS_PER_GPU = 8
DEFAULT_DEVICE = "cuda"
PYTHON_EXECUTABLE = "python"

# %%
# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def dict_to_cli_args(args_dict: Dict[str, Any]) -> str:
    """
    Convert configuration dictionary to OmegaConf-compatible CLI argument string.
    """
    parts = []
    for key, value in args_dict.items():
        if value is None:
            continue
        
        cli_key = key
        if isinstance(value, bool):
            parts.append(f"{cli_key}={str(value).lower()}")
        elif isinstance(value, list):
            formatted_elements = [
                f'"{item}"' if isinstance(item, str) else str(item) 
                for item in value
            ]
            list_str = f"[{','.join(formatted_elements)}]"
            parts.append(f"{cli_key}={shlex.quote(list_str)}")
        else:
            parts.append(f"{cli_key}={shlex.quote(str(value))}")
    
    return " ".join(parts)


def generate_job_name(params: Dict[str, Any]) -> str:
    """
    Generate short, descriptive SLURM job name from experiment parameters.
    
    """
    # Model name abbreviations for concise job names
    model_abbr = {
        "ShallowConvNet": "SCN", 
        "DeepConvNet": "DCN", 
        "EEGNetv4": "EEGNet", 
        "ATCNet": "ATC",
        "EEGNetv4S4": "EEGNetS4",
        "TEPNet": "TEPNet",
        "DeepTEPNet": "DeepTEPNet",
        "Ablation_NoS4": "Ablation_NoS4",
        "Ablation_ConvInsteadOfS4": "Ablation_ConvInsteadOfS4",
        "Ablation_S4_WithConvClassifier": "Ablation_S4_WithConvClassifier"
    }
    
    model_name = model_abbr.get(params.get("models_to_run")[0], "Model")
    ds_name = params["dataset_names"][0].replace("Lee2019_", "MI").replace("Yang2025", "Y25")
    config_tag = params.get("custom_config_tag", "Cfg")
    safe_config_tag = re.sub(r'[^a-zA-Z0-9]', '', config_tag)
    
    return f"{model_name}-{ds_name}-{safe_config_tag}"[:80]


def run_transfer_job(experiment_config: Dict[str, Any]) -> str:
    """
    Execute individual transfer learning experiment with proper environment setup.
    """
    exp_name = experiment_config.get("experiment_name", "default_exp")
    output_dir = Path(experiment_config.get("base_output_dir", BASE_OUTPUT_DIR / exp_name))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cli_args_str = dict_to_cli_args({k: v for k, v in experiment_config.items() if v is not None})
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    
    # Setup comprehensive logging
    log_file_dir = REPO_DIR / "job_logs"
    log_file_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_file_dir / f"transfer_log_{exp_name}_{job_id}.txt"

    # Construct execution command with environment setup
    cmd = f"""
set -e
echo "------ Environment Setup ------" > "{log_file}" && \
echo "Job ID: {job_id}" >> "{log_file}" && \
echo "Running on node: $(hostname)" >> "{log_file}" && \
echo "User: $(whoami)" >> "{log_file}" && \
echo "Conda Env: {CONDA_ENV}" >> "{log_file}" && \
echo "Activating Conda env..." >> "{log_file}" && \
source ~/.bashrc && \
conda activate {CONDA_ENV} && \
echo "Changing to repo directory..." >> "{log_file}" && \
cd "{REPO_DIR}" && \
echo "Setting MKL environment variables..." >> "{log_file}" && \
export MKL_THREADING_LAYER=GNU && \
echo "------ Running Command ------" >> "{log_file}" && \
echo "{PYTHON_EXECUTABLE} {SCRIPT_TO_RUN} {cli_args_str}" >> "{log_file}" && \
echo "------ Script Output ------" >> "{log_file}" && \
{PYTHON_EXECUTABLE} -u "{SCRIPT_TO_RUN}" {cli_args_str} 2>&1 | tee -a "{log_file}"
"""
    
    try:
        subprocess.run(cmd, shell=True, check=True, executable="/bin/bash", text=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}. Check log: {log_file}")
        raise RuntimeError(f"Script for '{exp_name}' failed. Log: {log_file}") from e
    
    return f"Finished: {exp_name}"

# %%
# =============================================================================
# EXPERIMENT CONFIGURATION DEFINITIONS
# =============================================================================

def get_alignment_configurations() -> List[Dict[str, Any]]:
    """
    Define alignment strategy configurations for evaluation.
    """
    return [
        # Baseline configurations (commented for selective execution)
        # {
        #     "config_name": "FM-None_A-Eucl_AdaBN-F", 
        #     "finetune_mode": "none", 
        #     "alignment_type": "euclidean", 
        #     "use_adabn": False
        # },
        # {
        #     "config_name": "FM-Dec_A-Eucl_AdaBN-F", 
        #     "finetune_mode": "decision_only", 
        #     "alignment_type": "euclidean", 
        #     "use_adabn": False
        # },
        
        # Active experimental configurations
        {
            "config_name": "FM-Full_A-None_AdaBN-F", 
            "finetune_mode": "full", 
            "alignment_type": "none", 
            "use_adabn": False
        },
        # {
        #      "config_name": "FM-Full_A-Eucl_AdaBN-F", 
        #      "finetune_mode": "full", 
        #      "alignment_type": "euclidean", 
        #      "use_adabn": False
        # },
        {
            "config_name": "FM-Dec_A-None_AdaBN-F", 
            "finetune_mode": "decision_only", 
            "alignment_type": "none", 
            "use_adabn": False
        },
        { 
            "config_name": "FM-DecThr_A-None_AdaBN-F", 
            "finetune_mode": "decision_criterion_only", 
            "alignment_type": "none", 
            "use_adabn": False
        },
        # { 
        #     "config_name": "FM-DecThr_A-Eucl_AdaBN-F", 
        #     "finetune_mode": "decision_criterion_only", 
        #     "alignment_type": "euclidean", 
        #     "use_adabn": False
        # },
    ]


def get_base_config() -> Dict[str, Any]:
    """
    Define shared base configuration parameters for all experiments.
    """
    return {
        # Cross-validation and experimental design
        "n_splits": 10,
        "device": DEFAULT_DEVICE,
        
        # Checkpoint and logging configuration
        "save_checkpoints": False,
        "save_results": True,
        "use_wandb": True,
        "wandb_project": "alignment_studies_extremes",
        "save_last_pretrained_checkpoint": True,
        "save_last_finetuned_checkpoint": True,
        
        # Training and optimization parameters
        "lr_finetune": 1e-4,
        "finetune_warmup_trials": 25,
        "finetune_epochs": 1,
        
        # Subject-specific calibration settings
        "use_subject_specific_calibration": True,
        "num_calibration_trials": 100,
        "lr_calibration": 0.0001,
        "calibration_epochs": 50,
        
        # Signal preprocessing parameters
        "tmin": -0.060,  # Start time relative to stimulus (seconds)
        "tmax": -0.010,  # End time relative to stimulus (seconds)
        "window_size": 50,
        
        # Classification and evaluation settings
        "use_binary_classification": True,
        "use_tta": True,  # Test-time augmentation
        "focal_loss_alpha": 3.0,
        "shuffle_test_labels": False,
        
        # Alignment-specific parameters
        "alignment_ref_ema_beta": 0.99,
        "ea_backrotation": True,
    }


def generate_experiment_configs() -> List[Dict[str, Any]]:
    """
    Generate complete experimental grid for comprehensive evaluation.
    """
    # Dataset configurations
    # Commented datasets available for extended evaluation:
    # datasets = [
    #     ["TMSEEGClassificationTEPfree"], 
    #     ["TMSEEGClassificationTEP"], 
    #     ["TMSEEGClassification"]
    # ]
    
    # Active dataset configurations
    datasets = [
        ["SICIClassification"], 
        ["SICFClassification"], 
        ["SinglePulseClassification"], 
    ]
    
    # Pre-training configuration
    pretrain_settings = [False]  # False = use pretrained models, True = train from scratch
    
    # Model architecture selection
    # Available models (commented for selective execution):
    # models_to_evaluate = [
    #     "PRIME", "Ablation_ConvInsteadOfS4", "Ablation_NoS4", 
    #     "Ablation_S4_WithConvClassifier", "EEGNetv4", "DeepConvNet", 
    #     "ShallowConvNet", "ATCNet"
    # ]
    
    # Currently active model for evaluation
    models_to_evaluate = ["PRIME"] 
    
    # Get alignment strategies and base configuration
    alignment_configurations = get_alignment_configurations()
    base_grid_dir = "10ms_pp_w50"  # Results subdirectory identifier
    global_base_config = get_base_config()

    # Generate all experiment combinations
    experiments = []
    
    for model, dataset, no_pretrain, align_conf in itertools.product(
        models_to_evaluate, datasets, pretrain_settings, alignment_configurations
    ):
        # Create experiment-specific configuration
        exp_config = global_base_config.copy()
        exp_config.update({k: v for k, v in align_conf.items() if k != "config_name"})
        
        # Add pre-training suffix for identification
        pretrain_suffix = "_no_pretrain" if no_pretrain else ""
        
        # Update with experiment-specific parameters
        exp_config.update({
            "models_to_run": [model],
            "dataset_names": dataset,
            "no_pretrain": no_pretrain,
            "custom_config_tag": align_conf['config_name'],
            "wandb_group": f"AlignFinal_{align_conf['config_name']}{pretrain_suffix}"
        })
        
        # Generate filesystem-safe experiment name
        raw_exp_name = f"AlignEval_{model}_{dataset[0]}_{align_conf['config_name']}{pretrain_suffix}"
        exp_config["experiment_name"] = re.sub(r'[^a-zA-Z0-9_.-]+', '', raw_exp_name)
        exp_config["base_output_dir"] = str(BASE_OUTPUT_DIR / base_grid_dir / exp_config["experiment_name"])
        
        experiments.append(exp_config)
    
    return experiments

# %%
# =============================================================================
# MAIN EXECUTION LOGIC
# =============================================================================

def main():
    """
    Main execution function for SLURM job submission system.
    """
    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Submit Transfer Learning Jobs for EEG Alignment Studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python script.py --dry-run    # Preview configurations without submission
    python script.py             # Submit all jobs to SLURM scheduler
        """
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Display experiment configurations without submitting SLURM jobs"
    )
    cli_args = parser.parse_args()
    
    # Generate comprehensive experiment grid
    experiments_to_run = generate_experiment_configs()
    total_jobs = len(experiments_to_run)
    
    print(f"Generated {total_jobs} experiment configurations for evaluation.")
    print(f"Target partition: {SLURM_PARTITION}")
    print(f"Resource allocation: {CPUS_PER_GPU} CPUs, {MEM_GB_PER_GPU}GB RAM per GPU")
    
    # Handle dry-run mode for configuration preview
    if cli_args.dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN MODE: Configuration Preview")
        print(f"{'='*60}")
        print(f"Total jobs that would be submitted: {total_jobs}")
        print("\nSample experiment configurations:")
        
        for i, cfg in enumerate(experiments_to_run[:3]):
            print(f"\n{'-'*40}")
            print(f"Example Configuration {i+1}/{min(3, total_jobs)}")
            print(f"{'-'*40}")
            for key, val in sorted(cfg.items()):
                print(f"  {key:<30}: {val}")
        
        if total_jobs > 3:
            print(f"\n... and {total_jobs - 3} additional configurations")
        
        print(f"\n{'='*60}")
        print("To submit jobs, run without --dry-run flag")
        print(f"{'='*60}")
        sys.exit(0)
    
    # Production job submission
    print(f"\n{'='*60}")
    print(f"SUBMITTING {total_jobs} JOBS TO SLURM")
    print(f"{'='*60}")
    
    submission_start_time = datetime.now()
    submitted_jobs = []
    
    for i, exp_config in enumerate(experiments_to_run):
        job_name = generate_job_name(exp_config)
        
        print(f"\n[{i+1:3d}/{total_jobs}] Submitting: {job_name}")
        print(f"         Experiment: {exp_config['experiment_name']}")
        
        # Configure logging directory with timestamp
        log_folder = REPO_DIR / "slurm_logs" / f"{datetime.now().strftime('%Y-%m-%d')}_{job_name}"
        executor = submitit.AutoExecutor(folder=str(log_folder))
        
        # Configure SLURM resource requirements
        executor.update_parameters(
            slurm_partition=SLURM_PARTITION,
            name=job_name,
            slurm_time="2-00:00:00",  # 2-day maximum runtime
            nodes=1,
            tasks_per_node=1,
            slurm_gpus_per_task=1,
            slurm_cpus_per_task=CPUS_PER_GPU,
            mem_gb=MEM_GB_PER_GPU,
        )
        
        # Submit job and track submission
        try:
            job = executor.submit(run_transfer_job, exp_config)
            submitted_jobs.append({
                'job_id': job.job_id,
                'name': job_name,
                'experiment': exp_config['experiment_name'],
                'log_folder': log_folder
            })
            print(f"         Job ID: {job.job_id}")
            print(f"         Logs: {log_folder}")
            
        except Exception as e:
            print(f"         ERROR: Failed to submit job - {str(e)}")
            continue
    
    # Submission summary
    submission_end_time = datetime.now()
    submission_duration = submission_end_time - submission_start_time
    
    print(f"\n{'='*60}")
    print("SUBMISSION COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully submitted: {len(submitted_jobs)}/{total_jobs} jobs")
    print(f"Submission duration: {submission_duration}")
    print(f"Estimated total GPU hours: {len(submitted_jobs) * 48}")  # 2 days max per job
    
    if len(submitted_jobs) < total_jobs:
        failed_count = total_jobs - len(submitted_jobs)
        print(f"WARNING: {failed_count} jobs failed to submit")
    
    print(f"\nMonitor job status with: squeue -u {os.environ.get('USER', 'user')}")
    print(f"Cancel all jobs with: scancel -u {os.environ.get('USER', 'user')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()