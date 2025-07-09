#!/usr/bin/env python3
"""
Diagnose what happened to missing SLURM jobs.
"""

import subprocess
import yaml
from yaml import Loader
from pathlib import Path
import re
from datetime import datetime, timedelta
import sys

def load_config():
    """Load configuration from configfile.yaml"""
    with open("configfile.yaml", 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    return config

def get_job_history():
    """Get job history from sacct"""
    try:
        # Get jobs from last 24 hours
        result = subprocess.run([
            'sacct', '-u', '$USER', '--format=JobID,JobName,State,ExitCode,Start,End,Elapsed',
            '--starttime=now-1day', '--parsable2'
        ], capture_output=True, text=True, check=True)
        
        lines = result.stdout.strip().split('\n')
        if len(lines) < 2:
            return []
        
        # Parse header
        header = lines[0].split('|')
        jobs = []
        
        for line in lines[1:]:
            if line.strip():
                parts = line.split('|')
                if len(parts) >= len(header):
                    job_data = dict(zip(header, parts))
                    # Only include our artificial data jobs
                    if job_data.get('JobName', '').startswith('artdata_'):
                        jobs.append(job_data)
        
        return jobs
    except subprocess.CalledProcessError as e:
        print(f"❌ Error getting job history: {e}")
        return []

def check_log_files():
    """Check log files for errors"""
    log_dir = Path("slurm_logs")
    if not log_dir.exists():
        return []
    
    log_files = list(log_dir.glob("slurm-*.out")) + list(log_dir.glob("slurm-*.err"))
    
    error_info = []
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
            # Look for errors
            if 'ERROR' in content or 'FAILED' in content or 'Error' in content:
                error_info.append({
                    'file': log_file.name,
                    'content': content[-1000:]  # Last 1000 chars
                })
        except Exception as e:
            print(f"⚠️  Could not read {log_file}: {e}")
    
    return error_info

def check_missing_combinations():
    """Check which combinations are missing and why"""
    config = load_config()
    sessions = config['sessions']
    epochs = config['epochs']
    trialtypes = config['trialtypes']
    processes = config['processes']
    
    missing_details = []
    
    for session in sessions:
        for epoch in epochs:
            for trialtype in trialtypes:
                combination = f"{session}/{epoch}/{trialtype}"
                
                # Check if input file exists
                input_path = Path(f'/users/bouss/spade/SPADE_surrogates/data/concatenated_spiketrains/{session}/{epoch}_{trialtype}.npy')
                input_exists = input_path.exists()
                
                # Check if output files exist
                output_status = {}
                
                if 'ppd' in processes:
                    ppd_path = Path(f'/users/bouss/spade/SPADE_surrogates/data/artificial_data/ppd/{session}/ppd_{epoch}_{trialtype}.npy')
                    output_status['ppd'] = ppd_path.exists()
                
                if 'gamma' in processes:
                    gamma_path = Path(f'/users/bouss/spade/SPADE_surrogates/data/artificial_data/gamma/{session}/gamma_{epoch}_{trialtype}.npy')
                    cv2_path = Path(f'/users/bouss/spade/SPADE_surrogates/data/artificial_data/gamma/{session}/cv2s_{epoch}_{trialtype}.npy')
                    output_status['gamma'] = gamma_path.exists()
                    output_status['cv2'] = cv2_path.exists()
                
                # If any output is missing, record details
                if not all(output_status.values()):
                    missing_details.append({
                        'combination': combination,
                        'input_exists': input_exists,
                        'output_status': output_status
                    })
    
    return missing_details

def analyze_job_failures(job_history):
    """Analyze job failures from history"""
    failed_jobs = []
    successful_jobs = []
    
    for job in job_history:
        state = job.get('State', '')
        exit_code = job.get('ExitCode', '')
        job_name = job.get('JobName', '')
        
        if state in ['FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY']:
            failed_jobs.append({
                'job_name': job_name,
                'state': state,
                'exit_code': exit_code,
                'start': job.get('Start', ''),
                'end': job.get('End', ''),
                'elapsed': job.get('Elapsed', '')
            })
        elif state == 'COMPLETED':
            successful_jobs.append(job)
    
    return failed_jobs, successful_jobs

def main():
    """Main diagnostic function"""
    print("="*80)
    print("DIAGNOSING MISSING JOBS")
    print("="*80)
    
    # 1. Check job history
    print("\n🔍 1. CHECKING JOB HISTORY...")
    job_history = get_job_history()
    failed_jobs = []
    successful_jobs = []
    
    if job_history:
        failed_jobs, successful_jobs = analyze_job_failures(job_history)
        
        print(f"📊 Job History Summary:")
        print(f"   Total jobs found: {len(job_history)}")
        print(f"   ✅ Successful: {len(successful_jobs)}")
        print(f"   ❌ Failed: {len(failed_jobs)}")
        
        if failed_jobs:
            print(f"\n❌ FAILED JOBS:")
            for job in failed_jobs[:10]:  # Show first 10
                print(f"   {job['job_name']}: {job['state']} (exit code: {job['exit_code']})")
            
            if len(failed_jobs) > 10:
                print(f"   ... and {len(failed_jobs) - 10} more")
        
        # Group failures by type
        failure_types = {}
        for job in failed_jobs:
            state = job['state']
            failure_types[state] = failure_types.get(state, 0) + 1
        
        if failure_types:
            print(f"\n📈 Failure Types:")
            for failure_type, count in failure_types.items():
                print(f"   {failure_type}: {count}")
    else:
        print("❌ No job history found (jobs might be too old or sacct not available)")
    
    # 2. Check log files
    print(f"\n🔍 2. CHECKING LOG FILES...")
    error_logs = check_log_files()
    
    if error_logs:
        print(f"📋 Found {len(error_logs)} log files with errors:")
        for i, error in enumerate(error_logs[:5]):  # Show first 5
            print(f"\n   📄 {error['file']}:")
            print(f"      {error['content'][:200]}...")
    else:
        print("✅ No error logs found or no errors in logs")
    
    # 3. Check missing combinations
    print(f"\n🔍 3. ANALYZING MISSING COMBINATIONS...")
    missing_details = check_missing_combinations()
    
    if missing_details:
        print(f"📊 Missing Combinations Analysis:")
        
        # Count by reason
        no_input = sum(1 for m in missing_details if not m['input_exists'])
        has_input = len(missing_details) - no_input
        
        print(f"   Total missing: {len(missing_details)}")
        print(f"   Missing input file: {no_input}")
        print(f"   Has input, missing output: {has_input}")
        
        # Show examples
        print(f"\n📋 Examples of missing combinations:")
        for missing in missing_details[:10]:
            combination = missing['combination']
            input_status = "✅" if missing['input_exists'] else "❌"
            output_status = ", ".join([f"{k}: {'✅' if v else '❌'}" for k, v in missing['output_status'].items()])
            print(f"   {combination}: Input {input_status}, Output [{output_status}]")
    
    # 4. Suggestions
    print(f"\n💡 SUGGESTIONS:")
    
    if failed_jobs:
        print(f"   1. Check specific log files in slurm_logs/ directory")
        print(f"   2. Look for common error patterns (memory, timeout, missing files)")
        print(f"   3. Consider resubmitting failed jobs with:")
        print(f"      - More memory (increase from 24500M)")
        print(f"      - More time (increase from 2:00:00)")
        print(f"      - Different partition")
    
    if missing_details:
        print(f"   4. For missing input files, check if preprocessing completed")
        print(f"   5. For missing outputs with existing inputs, resubmit those specific jobs")
    
    print(f"\n🔧 USEFUL COMMANDS:")
    print(f"   Check recent failed jobs: sacct -u $USER --state=FAILED --starttime=now-1day")
    print(f"   Check job details: sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Start,End,Elapsed")
    print(f"   Check system status: sinfo")
    print(f"   Check your limits: sacctmgr show associations user=$USER")

if __name__ == '__main__':
    main()