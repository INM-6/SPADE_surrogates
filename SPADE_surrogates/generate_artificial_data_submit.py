#!/usr/bin/env python3
"""
Parallel submission script for artificial data generation.
This script submits multiple SLURM jobs for each session/epoch/trialtype combination.
"""

import yaml
from yaml import Loader
import subprocess
import sys
from pathlib import Path
import time

def load_config():
    """Load configuration from configfile.yaml"""
    with open("configfile.yaml", 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    return config

def submit_job(session, epoch, trialtype, job_name, log_dir, time_limit="2:00:00", memory="24500M"):
    """Submit a single SLURM job"""
    
    # Create SLURM script content
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=blaustein
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time={time_limit}
#SBATCH --mem={memory}
#SBATCH --output={log_dir}/slurm-%j.out
#SBATCH --error={log_dir}/slurm-%j.err

# Load environment
source ~/.bashrc
cd "$SLURM_SUBMIT_DIR"
source ~/spade_env/bin/activate

# Run the single job script
echo "Starting job for {session}/{epoch}/{trialtype}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Started at: $(date)"

python generate_artificial_data.py {session} {epoch} {trialtype}

echo "Finished at: $(date)"
"""
    
    # Write SLURM script to temporary file
    script_path = f"/tmp/slurm_job_{session}_{epoch}_{trialtype}.sh"
    with open(script_path, 'w') as f:
        f.write(slurm_script)
    
    # Submit job
    try:
        result = subprocess.run(['sbatch', script_path], 
                              capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        print(f"✅ Submitted job {job_id}: {session}/{epoch}/{trialtype}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to submit job for {session}/{epoch}/{trialtype}: {e}")
        return None
    finally:
        # Clean up temporary script
        try:
            Path(script_path).unlink()
        except:
            pass

def main():
    """Main function to submit all jobs"""
    # Load configuration
    config = load_config()
    
    sessions = config['sessions']
    epochs = config['epochs']
    trialtypes = config['trialtypes']
    
    # Calculate total number of jobs
    total_jobs = len(sessions) * len(epochs) * len(trialtypes)
    
    print("="*80)
    print("PARALLEL ARTIFICIAL DATA GENERATION - JOB SUBMISSION")
    print("="*80)
    print(f"Sessions: {len(sessions)} {sessions}")
    print(f"Epochs: {len(epochs)} {epochs}")
    print(f"Trial types: {len(trialtypes)} {trialtypes}")
    print(f"Total jobs to submit: {total_jobs}")
    print("="*80)
    
    # Create logs directory
    log_dir = Path("slurm_logs")
    log_dir.mkdir(exist_ok=True)
    print(f"📁 Log directory: {log_dir}")
    
    # Ask for confirmation
    response = input(f"\n🚀 Ready to submit {total_jobs} jobs? (y/N): ")
    if response.lower() != 'y':
        print("❌ Cancelled by user")
        sys.exit(0)
    
    # Submit jobs
    submitted_jobs = []
    failed_submissions = []
    
    print(f"\n📤 Starting job submission...")
    
    for session in sessions:
        for epoch in epochs:
            for trialtype in trialtypes:
                job_name = f"artdata_{session}_{epoch}_{trialtype}"
                
                job_id = submit_job(
                    session=session,
                    epoch=epoch, 
                    trialtype=trialtype,
                    job_name=job_name,
                    log_dir=log_dir,
                    time_limit="2:00:00",  # Adjust as needed
                    memory="24500M"        # Adjust as needed
                )
                
                if job_id:
                    submitted_jobs.append((job_id, session, epoch, trialtype))
                else:
                    failed_submissions.append((session, epoch, trialtype))
                
                # Small delay to avoid overwhelming the scheduler
                time.sleep(0.1)
    
    # Summary
    print(f"\n" + "="*80)
    print("JOB SUBMISSION SUMMARY")
    print("="*80)
    print(f"✅ Successfully submitted: {len(submitted_jobs)} jobs")
    print(f"❌ Failed submissions: {len(failed_submissions)} jobs")
    
    if failed_submissions:
        print(f"\n❌ Failed submissions:")
        for session, epoch, trialtype in failed_submissions:
            print(f"   - {session}/{epoch}/{trialtype}")
    
    if submitted_jobs:
        print(f"\n📊 Submitted job IDs:")
        for job_id, session, epoch, trialtype in submitted_jobs:
            print(f"   {job_id}: {session}/{epoch}/{trialtype}")
    
    # Provide monitoring commands
    print(f"\n📋 Monitoring commands:")
    print(f"   Check job status: squeue -u $USER")
    print(f"   Check specific jobs: squeue -j {','.join([job[0] for job in submitted_jobs[:5]])}...")
    print(f"   Cancel all jobs: scancel -u $USER")
    print(f"   View logs: ls -la {log_dir}/")
    
    print(f"\n🎉 All jobs submitted! Monitor with 'squeue -u $USER'")

if __name__ == '__main__':
    main()