#!/usr/bin/env bash
# Script to run the GPU testing Snakemake workflow
# This tests Cellpose-SAM model loading across all GPU nodes in the cluster

set -euo pipefail

echo "=========================================="
echo "GPU Cellpose-SAM Testing Workflow"
echo "=========================================="
echo "Date: $(date --iso-8601=seconds)"
echo ""

# Ensure we're in the gpu_test directory
cd "$(dirname "$0")"

echo "Working directory: $(pwd)"
echo ""

# Check if conda environment is activated
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "WARNING: No conda environment appears to be activated!"
    echo "Please activate your test environment first:"
    echo "  conda activate <your_test_env>"
    exit 1
fi

echo "Active conda environment: ${CONDA_DEFAULT_ENV}"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Check if snakemake is available
if ! command -v snakemake &> /dev/null; then
    echo "ERROR: snakemake command not found!"
    echo "Please ensure snakemake is installed in your conda environment."
    exit 1
fi

echo "Snakemake version: $(snakemake --version)"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs/slurm

echo "=========================================="
echo "Launching Snakemake workflow"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Config file: config.yaml"
echo "  - Executor: SLURM"
echo "  - Parallel jobs: 24 (one per GPU node)"
echo "  - SLURM logs: logs/slurm/"
echo "  - Results: results/"
echo ""

# Run snakemake with SLURM executor
# --jobs 24: Allow up to 24 parallel jobs (one per GPU node)
# --executor slurm: Use SLURM executor for cluster submission
# --configfile: Use our config.yaml
# --latency-wait: Wait 10s for files to appear (network filesystem)
# --rerun-incomplete: Rerun any incomplete jobs from previous runs
# --printshellcmds: Print shell commands being executed
# --slurm-logdir: Directory for SLURM stdout/stderr logs
# --default-resources: Default SLURM resources (will be overridden by config)

snakemake \
    --snakefile Snakefile \
    --configfile config.yaml \
    --jobs 24 \
    --executor slurm \
    --latency-wait 10 \
    --rerun-incomplete \
    --printshellcmds \
    --slurm-logdir logs/slurm \
    --default-resources slurm_partition=gpu mem_mb=8000 runtime=5

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Workflow completed successfully!"
    echo "=========================================="
    echo ""
    echo "Results saved to: results/"
    echo ""
    echo "View summary:"
    echo "  cat results/summary.txt"
    echo ""
    echo "View individual results:"
    echo "  ls results/*.json"
    echo ""
    echo "View SLURM logs:"
    echo "  ls logs/slurm/"
else
    echo "✗ Workflow failed with exit code: $EXIT_CODE"
    echo "=========================================="
    echo ""
    echo "Check logs for errors:"
    echo "  ls logs/slurm/"
fi
echo ""

exit $EXIT_CODE
