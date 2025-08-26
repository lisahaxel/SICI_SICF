import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import logging

import submitit
logging.basicConfig(level=logging.DEBUG)

# %%
# --- Core Configuration ---

# Path to repository and the script
USER = os.environ.get('USER', 'user')
REPO_DIR = Path(f"/mnt/lustre/work/macke/{USER}/repos/eegjepa")
SCRIPT_TO_RUN = REPO_DIR / "EDAPT_neurips/EDAPT_TMS/SICISICF/preprocessing/preprocessing_single_subject.py" 
#SCRIPT_TO_RUN = REPO_DIR / "EDAPT_neurips/EDAPT_TMS/SICISICF/preprocessing/preprocessing_single_subject_offline.py" 

# Data and Logging
# This now points to the directory containing your SICI-SICF_sub-XX-epo.fif files
DATA_SOURCE_DIR = REPO_DIR / "EDAPT_neurips/EDAPT_TMS/SICISICF/processed_data"
BASE_LOG_DIR = REPO_DIR / "slurm_logs_preprocessing"

# SLURM Configuration for CPU jobs
# IMPORTANT: Verify the partition name. 'cpu-only' or 'compute' are common.
SLURM_PARTITION = "cpu-galvani" 
MEM_GB_PER_JOB = 64  # Memory per job
CPUS_PER_JOB = 1     # IMPORTANT: Use 1 CPU to ensure sequential processing and accurate latency measurement
SUBJECTS_PER_JOB = 1 # As requested

# %%
# --- Utility and Job Execution Functions ---

def discover_subjects(data_dir: Path) -> List[str]:
    """Scans the data directory and returns a list of unique subject ID prefixes."""
    subjects = set()
    if not data_dir.is_dir():
        print(f"ERROR: Data directory not found at {data_dir}")
        return []
        
    for f in data_dir.glob("*-epo.fif"):
        # Extracts 'SICI-SICF_sub-01' from 'SICI-SICF_sub-01-epo.fif'
        subject_id = f.stem.replace("-epo", "")
        subjects.add(subject_id)
        
    return sorted(list(subjects))

def chunk_list(data: list, size: int):
    """Yields successive n-sized chunks from a list."""
    for i in range(0, len(data), size):
        yield data[i:i + size]

def run_sequential_preprocessing(subject_chunk: List[str]):
    """
    Executes the preprocessing script sequentially for a chunk of subjects.
    This function runs inside the SLURM job.
    """
    print(f"--- SLURM Job {os.environ.get('SLURM_JOB_ID')} starting ---")
    print(f"Processing {len(subject_chunk)} subjects in this job.")

    # This is the directory containing your preprocessing_single_subject.py,
    # ica_calibrator.py, and channel_interpolations.py
    script_dir = SCRIPT_TO_RUN.parent

    for i, subject in enumerate(subject_chunk):
        print(f"\n[{i+1}/{len(subject_chunk)}] Preparing to process: {subject}")
        
        # Construct the simplified command
        command = [
            "python", "-u", str(SCRIPT_TO_RUN),
            "--subject", subject
        ]
        print(f"  - Command to run: {' '.join(command)}")
        
        # The environment setup is good, but we will add 'cwd' for robustness
        job_env = os.environ.copy()
        job_env["PYTHONPATH"] = f"{script_dir}{os.pathsep}{job_env.get('PYTHONPATH', '')}"
        
        try:
            print("  - Executing subprocess...")
            subprocess.run(
                command, 
                check=True,
                text=True,
                env=job_env,
                stderr=subprocess.STDOUT,
                cwd=script_dir
            )
            print(f"  - SUCCESSFULLY finished processing {subject}")

        except subprocess.CalledProcessError as e:
            print(f"---  ERROR: Subprocess failed for subject {subject} with exit code {e.returncode}.  ---")
            print("---  The Python traceback from the script should be visible above this line.  ---")
            # If one subject fails, it's often better to continue with the rest
            continue
    
    return f"Finished job {os.environ.get('SLURM_JOB_ID')}, processed {len(subject_chunk)} subjects."

# %%
# --- Main Execution ---

def main():
    """Main function to discover subjects and submit SLURM jobs."""
    parser = argparse.ArgumentParser(description="Submit sequential preprocessing jobs to SLURM.")
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Print job configurations without submitting."
    )
    cli_args = parser.parse_args()

    # Discover and chunk subjects
    all_subjects = discover_subjects(DATA_SOURCE_DIR)
    if not all_subjects:
        print("No subjects found. Exiting.")
        return
    
    subject_chunks = list(chunk_list(all_subjects, SUBJECTS_PER_JOB))
    total_jobs = len(subject_chunks)
    
    print(f"Discovered {len(all_subjects)} subjects.")
    print(f"Grouping into {total_jobs} jobs with up to {SUBJECTS_PER_JOB} subjects each.")

    if cli_args.dry_run:
        print("\n--- DRY RUN: Job Configurations ---")
        for i, chunk in enumerate(subject_chunks[:3]):
            print(f"\nJob {i+1}/{total_jobs} would process:")
            for subject in chunk:
                print(f"  - {subject}")
        sys.exit(0)


    # Submit jobs
    print(f"\nSubmitting {total_jobs} jobs to SLURM...")
    
    # Setup a general executor for all jobs
    date_str = datetime.now().strftime('%Y-%m-%d')
    log_folder = BASE_LOG_DIR / f"{date_str}_preprocessing_run_v2"
    executor = submitit.AutoExecutor(folder=str(log_folder))
    
    executor.update_parameters(
        slurm_partition=SLURM_PARTITION,
        slurm_time="2-00:00:00",  # 2 days
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=CPUS_PER_JOB,
        mem_gb=MEM_GB_PER_JOB,
    )
    
    jobs = []
    with executor.batch():
        for i, chunk in enumerate(subject_chunks):
            job_name = f"preprocess_batch_{i+1}"
            job = executor.submit(run_sequential_preprocessing, chunk)
            jobs.append(job)
            # This print statement no longer tries to access the job ID
            print(f"  Queueing Job {i+1}/{total_jobs}: {job_name}")

    # This new block runs AFTER all jobs have been submitted
    print(f"\nAll {total_jobs} jobs submitted successfully. Job IDs:")
    for job in jobs:
        print(f"  - {job.job_id}")
    
    print(f"\nSLURM logs will be stored in: {log_folder}")

if __name__ == "__main__":
    main()