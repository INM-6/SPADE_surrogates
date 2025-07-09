#!/usr/bin/env python3
"""
Monitor the progress of parallel artificial data generation jobs.
"""

import subprocess
import time
import yaml
from yaml import Loader
from pathlib import Path
import sys

def load_config():
    """Load configuration from configfile.yaml"""
    with open("configfile.yaml", 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    return config

def get_job_status():
    """Get current job status from SLURM"""
    try:
        result = subprocess.run(['squeue', '-u', '$USER', '--format=%i,%T,%j'], 
                              capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')[1:]  # Skip header
        
        jobs = []
        for line in lines:
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 3:
                    job_id = parts[0]
                    status = parts[1]
                    job_name = parts[2]
                    if job_name.startswith('artdata_'):
                        jobs.append((job_id, status, job_name))
        
        return jobs
    except subprocess.CalledProcessError:
        return []

def check_output_files():
    """Check which output files have been created"""
    config = load_config()
    sessions = config['sessions']
    epochs = config['epochs']
    trialtypes = config['trialtypes']
    processes = config['processes']
    
    completed_combinations = []
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
                
                if files_exist:
                    completed_combinations.append((session, epoch, trialtype))
                else:
                    missing_combinations.append((session, epoch, trialtype))
    
    return completed_combinations, missing_combinations

def main():
    """Main monitoring function"""
    config = load_config()
    total_combinations = len(config['sessions']) * len(config['epochs']) * len(config['trialtypes'])
    
    print("="*80)
    print("PARALLEL ARTIFICIAL DATA GENERATION - MONITORING")
    print("="*80)
    
    try:
        while True:
            # Get job status
            jobs = get_job_status()
            
            # Count by status
            status_counts = {}
            for job_id, status, job_name in jobs:
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Check completed files
            completed, missing = check_output_files()
            
            # Display status
            print(f"\n📊 STATUS UPDATE - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"─" * 50)
            
            if jobs:
                print(f"🔄 SLURM Jobs:")
                for status, count in sorted(status_counts.items()):
                    print(f"   {status}: {count}")
                print(f"   Total active jobs: {len(jobs)}")
            else:
                print(f"🔄 No active SLURM jobs found")
            
            print(f"\n📁 Output Files:")
            print(f"   ✅ Completed: {len(completed)}/{total_combinations} ({len(completed)/total_combinations*100:.1f}%)")
            print(f"   ⏳ Missing: {len(missing)}")
            
            if len(completed) == total_combinations:
                print(f"\n🎉 ALL JOBS COMPLETED! 🎉")
                break
            
            # Show some recent completions
            if completed:
                print(f"\n📋 Recent completions (last 5):")
                for session, epoch, trialtype in completed[-5:]:
                    print(f"   ✅ {session}/{epoch}/{trialtype}")
            
            # Show some pending jobs
            if missing:
                print(f"\n⏳ Still pending (showing first 5):")
                for session, epoch, trialtype in missing[:5]:
                    print(f"   ⏳ {session}/{epoch}/{trialtype}")
            
            # Wait before next check
            print(f"\n⏰ Next check in 30 seconds... (Ctrl+C to exit)")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print(f"\n\n👋 Monitoring stopped by user")
        
        # Final summary
        completed, missing = check_output_files()
        print(f"\n📊 FINAL STATUS:")
        print(f"   ✅ Completed: {len(completed)}/{total_combinations}")
        print(f"   ⏳ Missing: {len(missing)}")
        
        if missing:
            print(f"\n❌ Missing combinations:")
            for session, epoch, trialtype in missing:
                print(f"   - {session}/{epoch}/{trialtype}")

if __name__ == '__main__':
    main()