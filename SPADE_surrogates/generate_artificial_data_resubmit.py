#!/usr/bin/env python3
"""
Resubmit only the missing jobs (combinations that don't have output files).
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

def find_missing_combinations():
    """Find combinations that don't have output files"""
    config = load_config()
    sessions = config['sessions']
    epochs = config['epochs']
    trialtypes = config['trialtypes']
    processes = config['processes']
    
    missing_combinations = []
    
    for session in sessions:
        for epoch in epochs:
            for trialtype in trialtypes:
                # Check if all required output files exist
                files_exist = True
                
                if 'ppd' in processes:
                    ppd_path = Path(f'/users/bouss/spade/SPADE_surrogates/data/artificial_data/ppd/{session}/ppd_{epoch}_{trialtype}.npy')
                    if not ppd_path.exists():
                        files_exist = False
                
                if 'gamma' in processes:
                    gamma_path = Path(f'/users/bouss/spade/SPADE_surrogates/data/artificial_data/gamma/{session}/gamma_{epoch}_{trialtype}.npy')
                    cv2_path = Path(f'/users/bouss/spade/SPADE_surrogates/data/artificial_data/gamma/{session}/cv2s_{epoch}_{trialtype}.npy')
                    if not gamma_path.exists() or not cv2_path.exists():
                        files_exist = False
                
                if not files_exist:
                    # Also check if input file exists
                    input_path = Path(f'/users/bouss/spade/SPADE_surrogates/data/concatenated_spiketrains/{session}/{epoch}_{trialtype}.npy')
                    if input_path.exists():
                        missing_combinations.append((session, epoch, trialtype))
                    else:
                        print(f"⚠️  Skipping {session}/{epoch}/{trialtype} - no input file")
    
    return missing_combinations

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
echo "Starting RESUBMITTED job for {session}/{epoch}/{trialtype}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Started at: $(date)"

python generate_artificial_data.py {session} {epoch} {trialtype}

echo "Finished at: $(date)"
"""
    
    # Write SLURM script to temporary file
    script_path = f"/tmp/slurm_resubmit_{session}_{epoch}_{trialtype}.sh"
    with open(script_path, 'w') as f:
        f.write(slurm_script)
    
    # Submit job
    try:
        result = subprocess.run(['sbatch', script_path], 
                              capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        print(f"✅ Resubmitted job {job_id}: {session}/{epoch}/{trialtype}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to resubmit job for {session}/{epoch}/{trialtype}: {e}")
        return None
    finally:
        # Clean up temporary script
        try:
            Path(script_path).unlink()
        except:
            pass

def main():
    """Main function to resubmit missing jobs"""
    print("="*80)
    print("RESUBMITTING MISSING JOBS")
    print("="*80)
    
    # Find missing combinations
    print("🔍 Finding missing combinations...")
    missing_combinations = find_missing_combinations()
    
    if not missing_combinations:
        print("🎉 No missing combinations found! All jobs completed successfully.")
        return
    
    print(f"📊 Found {len(missing_combinations)} missing combinations to resubmit")
    
    # Show some examples
    print(f"\n📋 Examples of missing combinations:")
    for i, (session, epoch, trialtype) in enumerate(missing_combinations[:10]):
        print(f"   {i+1}. {session}/{epoch}/{trialtype}")
    
    if len(missing_combinations) > 10:
        print(f"   ... and {len(missing_combinations) - 10} more")
    
    # Create logs directory
    log_dir = Path("slurm_logs")
    log_dir.mkdir(exist_ok=True)
    print(f"\n📁 Log directory: {log_dir}")
    
    # Ask for confirmation
    response = input(f"\n🚀 Ready to resubmit {len(missing_combinations)} missing jobs? (y/N): ")
    if response.lower() != 'y':
        print("❌ Cancelled by user")
        sys.exit(0)
    
    # Submit jobs
    submitted_jobs = []
    failed_submissions = []
    
    print(f"\n📤 Starting job resubmission...")
    
    for session, epoch, trialtype in missing_combinations:
        job_name = f"resubmit_{session}_{epoch}_{trialtype}"
        
        job_id = submit_job(
            session=session,
            epoch=epoch, 
            trialtype=trialtype,
            job_name=job_name,
            log_dir=log_dir,
            time_limit="2:00:00",  # You might want to increase this
            memory="24500M"        # You might want to increase this
        )
        
        if job_id:
            submitted_jobs.append((job_id, session, epoch, trialtype))
        else:
            failed_submissions.append((session, epoch, trialtype))
        
        # Small delay to avoid overwhelming the scheduler
        time.sleep(0.1)
    
    # Summary
    print(f"\n" + "="*80)
    print("RESUBMISSION SUMMARY")
    print("="*80)
    print(f"✅ Successfully resubmitted: {len(submitted_jobs)} jobs")
    print(f"❌ Failed resubmissions: {len(failed_submissions)} jobs")
    
    if failed_submissions:
        print(f"\n❌ Failed resubmissions:")
        for session, epoch, trialtype in failed_submissions:
            print(f"   - {session}/{epoch}/{trialtype}")
    
    # Provide monitoring commands
    print(f"\n📋 Monitoring commands:")
    print(f"   Check job status: squeue -u $USER")
    print(f"   Monitor progress: python monitor_jobs.py")
    print(f"   Cancel all jobs: scancel -u $USER")
    
    print(f"\n🎉 Resubmission complete! Monitor with 'squeue -u $USER'")

if __name__ == '__main__':
    main()