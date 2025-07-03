#!/usr/bin/env bash

#SBATCH --job-name=spade_snakemake
#SBATCH --output=logs/snakemake-%j.out
#SBATCH --error=logs/snakemake-%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=blaustein

# =============================================================================
# SPADE Analysis SLURM Submission Script
# =============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths (adjust these to your environment)
VIRTUAL_ENV_PATH="~/spade_env/bin/activate"
CONFIG_FILE="../configfile.yaml" 
CLUSTER_CONFIG="../cluster.json"
LOG_DIR="logs"
SLURM_LOG_DIR="slurm"

# Snakemake settings - optimized for cluster
MAX_JOBS=60   # Conservative limit to avoid overwhelming scheduler
CORES=240     # Total cores across multiple partitions (blaustein: 30×48=1440, hamstein: 16×128=2048)
LATENCY_WAIT=120  # Increased wait time for cluster file systems

# Create log directories if they don't exist
mkdir -p "${LOG_DIR}"
mkdir -p "${SLURM_LOG_DIR}"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

echo "=== SPADE Analysis Snakemake Submission ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Started at: $(date)"
echo "Working directory: $(pwd)"
echo "Configuration file: ${CONFIG_FILE}"
echo ""

# Activate virtual environment
echo "Activating virtual environment: ${VIRTUAL_ENV_PATH}"
source "${VIRTUAL_ENV_PATH}"

# Verify environment
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "Snakemake version: $(snakemake --version)"
echo "MPI version: $(mpirun --version | head -1)"
echo ""

# =============================================================================
# SNAKEMAKE EXECUTION
# =============================================================================

echo "=== Phase 1: Unlocking workflow ==="
snakemake --unlock \
    --configfile "${CONFIG_FILE}" \
    --cores 1

echo ""
echo "=== Phase 2: Executing workflow ==="
echo "Maximum concurrent jobs: ${MAX_JOBS}"
echo "Total cores available: ${CORES}"
echo ""

# Main snakemake execution with cluster submission
snakemake \
    --jobs "${MAX_JOBS}" \
    --cluster "sbatch \
        --ntasks={cluster.ntasks} \
        --cpus-per-task={cluster.cpus_per_task} \
        --time={cluster.time} \
        --mem={cluster.mem} \
        --partition={cluster.partition} \
        --job-name={cluster.jobname} \
        --output=${SLURM_LOG_DIR}/rule-%j.out \
        --error=${SLURM_LOG_DIR}/rule-%j.err \
        --mail-type=FAIL" \
    --cluster-config "${CLUSTER_CONFIG}" \
    --jobname "{jobid}.{rulename}" \
    --latency-wait "${LATENCY_WAIT}" \
    --keep-going \
    --rerun-incomplete \
    --cores "${CORES}" \
    --configfile "${CONFIG_FILE}"

# =============================================================================
# COMPLETION
# =============================================================================

EXIT_CODE=$?

echo ""
echo "=== Workflow Completion ==="
echo "Finished at: $(date)"
echo "Exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Workflow completed successfully!"
else
    echo "Workflow failed with exit code ${EXIT_CODE}"
fi

exit ${EXIT_CODE}