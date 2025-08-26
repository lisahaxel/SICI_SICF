import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List
import logging

import submitit
logging.basicConfig(level=logging.DEBUG)

# %%
# --- Core Configuration ---

# Path to repository and the script
USER = os.environ.get('USER', 'user')
REPO_DIR = Path(f"/mnt/lustre/work/macke/{USER}/repos/eegjepa")
SCRIPT_TO_RUN = REPO_DIR / "EDAPT_neurips/EDAPT_TMS/SICISICF/preprocessing/get_features.py"

# Data and Logging
# This now points to the directory containing your processed data from preprocessing
DATA_SOURCE_DIR = REPO_DIR / "EDAPT_neurips/EDAPT_TMS/SICISICF/data_processed_final_pre_ica_True_final_offline"
BASE_LOG_DIR = REPO_DIR / "slurm_logs_feature_extraction"

# SLURM Configuration for CPU jobs
SLURM_PARTITION = "cpu-galvani"  # Adjust to your cluster's partition name
MEM_GB_PER_JOB = 64 # Memory per job 
CPUS_PER_JOB = 1     # Use 1 CPU for sequential processing per job
SUBJECTS_PER_JOB = 1 # Process one subject per job for maximum parallelization

# %%
# --- Utility and Job Execution Functions ---

def discover_subjects(data_dir: Path) -> List[str]:
    """Scans the data directory and returns a list of unique subject ID prefixes."""
    subjects = set()
    if not data_dir.is_dir():
        print(f"ERROR: Data directory not found at {data_dir}")
        return []
        
    # Look for subject directories (e.g., SICI-SICF_sub-01, SICI-SICF_sub-02, etc.)
    for subject_dir in data_dir.iterdir():
        if subject_dir.is_dir() and "sub-" in subject_dir.name:
            subjects.add(subject_dir.name)
        
    return sorted(list(subjects))

def chunk_list(data: list, size: int):
    """Yields successive n-sized chunks from a list."""
    for i in range(0, len(data), size):
        yield data[i:i + size]

def run_sequential_feature_extraction(subject_chunk: List[str]):
    """
    Executes the feature extraction script sequentially for a chunk of subjects.
    This function runs inside the SLURM job.
    """
    print(f"--- SLURM Job {os.environ.get('SLURM_JOB_ID')} starting feature extraction ---")
    print(f"Processing {len(subject_chunk)} subjects in this job.")

    # This is the directory containing your get_features.py script
    script_dir = SCRIPT_TO_RUN.parent

    for i, subject in enumerate(subject_chunk):
        print(f"\n[{i+1}/{len(subject_chunk)}] Preparing to extract features for: {subject}")
        
        # Construct the command
        command = [
            "python", "-u", str(SCRIPT_TO_RUN),
            "--subject", subject
        ]
        print(f"  - Command to run: {' '.join(command)}")
        
        # Set up environment
        job_env = os.environ.copy()
        job_env["PYTHONPATH"] = f"{script_dir}{os.pathsep}{job_env.get('PYTHONPATH', '')}"
        
        try:
            print("  - Executing subprocess...")
            result = subprocess.run(
                command, 
                check=True,
                text=True,
                env=job_env,
                capture_output=True,  # Capture both stdout and stderr
                cwd=script_dir
            )
            
            # Print the output for debugging
            if result.stdout:
                print(f"  - STDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"  - STDERR:\n{result.stderr}")
                
            print(f"  - SUCCESSFULLY finished feature extraction for {subject}")

        except subprocess.CalledProcessError as e:
            print(f"---  ERROR: Subprocess failed for subject {subject} with exit code {e.returncode}  ---")
            print(f"---  STDOUT: {e.stdout}  ---")
            print(f"---  STDERR: {e.stderr}  ---")
            # Continue with other subjects even if one fails
            continue
        except Exception as e:
            print(f"---  UNEXPECTED ERROR for subject {subject}: {e}  ---")
            continue
    
    return f"Finished feature extraction job {os.environ.get('SLURM_JOB_ID')}, processed {len(subject_chunk)} subjects."

# %%
# --- Main Execution ---

def main():
    """Main function to discover subjects and submit SLURM jobs."""
    parser = argparse.ArgumentParser(description="Submit parallel feature extraction jobs to SLURM.")
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
        for i, chunk in enumerate(subject_chunks[:5]):  # Show first 5 jobs
            print(f"\nJob {i+1}/{total_jobs} would process:")
            for subject in chunk:
                print(f"  - {subject}")
        if total_jobs > 5:
            print(f"\n... and {total_jobs - 5} more jobs")
        sys.exit(0)

    # Submit jobs
    print(f"\nSubmitting {total_jobs} jobs to SLURM...")
    
    # Setup executor for all jobs
    date_str = datetime.now().strftime('%Y-%m-%d')
    log_folder = BASE_LOG_DIR / f"{date_str}_feature_extraction_run"
    executor = submitit.AutoExecutor(folder=str(log_folder))
    
    executor.update_parameters(
        slurm_partition=SLURM_PARTITION,
        slurm_time="1-00:00:00",  # 1 day should be plenty for feature extraction
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=CPUS_PER_JOB,
        mem_gb=MEM_GB_PER_JOB,
    )
    
    jobs = []
    with executor.batch():
        for i, chunk in enumerate(subject_chunks):
            job_name = f"features_batch_{i+1}"
            job = executor.submit(run_sequential_feature_extraction, chunk)
            jobs.append(job)
            print(f"  Queueing Job {i+1}/{total_jobs}: {job_name}")

    # Print job IDs after submission
    print(f"\nAll {total_jobs} jobs submitted successfully. Job IDs:")
    for i, job in enumerate(jobs):
        print(f"  - Job {i+1}: {job.job_id}")
    
    print(f"\nSLURM logs will be stored in: {log_folder}")

if __name__ == "__main__":
    main()