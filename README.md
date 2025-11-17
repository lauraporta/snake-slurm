# GPU Testing Workflow for Cellpose-SAM

This directory contains a Snakemake workflow to test Cellpose-SAM model loading and inference across all GPU nodes in the cluster.

**Architecture:** This workflow uses **Snakemake with the SLURM executor**, NOT direct sbatch submission. Snakemake handles job submission, dependency tracking, and parallelization automatically.

## Purpose

This workflow helps identify:
- Which GPU nodes have working CUDA installations
- Which nodes can successfully load the Cellpose-SAM model on GPU
- Which nodes can run inference successfully
- Any environment or configuration issues specific to certain nodes

## Prerequisites

1. Activate the `pm` conda environment:
   ```bash
   conda activate pm
   ```

2. Ensure you're in the `gpu_test` directory:
   ```bash
   cd gpu_test
   ```

## Usage

### Run the full test across all GPU nodes:

```bash
./run_gpu_test.sh
```

This will:
- Use Snakemake to orchestrate the workflow
- Submit one SLURM job per GPU node via Snakemake's SLURM executor (24 nodes total)
- Each job runs `test_gpu_node.py` to test PyTorch CUDA, model loading, and inference
- Aggregate results using `summarize_results.py`
- Save results to `results/` directory
- Generate a summary report

### Analyze the results:

```bash
./analyze_results.sh
```

This provides:
- Summary statistics (success/failure counts)
- Detailed breakdown by test category
- List of problematic nodes with error details
- Recommendations for fixing issues

### View summary report:

```bash
cat results/summary.txt
```

### View individual node results:

```bash
cat results/gpu-350-01_result.json | python -m json.tool
```

## Files

### Core Workflow Files
- `Snakefile` - Snakemake workflow definition (orchestrates GPU testing via SLURM executor)
- `config.yaml` - SLURM resource configuration (partition, memory, runtime, etc.)
- `run_gpu_test.sh` - Launcher script that invokes Snakemake

### Python Scripts (called by Snakemake rules)
- `test_gpu_node.py` - Tests a single GPU node (PyTorch, CUDA, model loading, inference)
- `summarize_results.py` - Aggregates individual test results into summary report
- `analyze_gpu_results.py` - Optional detailed analysis tool for deeper investigation

### Utility Files
- `analyze_results.sh` - Shell wrapper for running detailed analysis
- `discover_nodes.sh` - **Optional utility** to discover available GPU nodes (helpful when updating the node list in Snakefile)
- `logs/` - Directory for workflow and SLURM logs
- `results/` - Directory for test results (JSON files per node + summary)

## Node List

The workflow tests these GPU nodes across different partitions (edit `Snakefile` to customize):

**GPU partition:**
- gpu-350 series: 01-05
- gpu-380 series: 10-18
- gpu-sr670 series: 20-23
- gpu-sr675: 31

**GPU_LOWP partition:**
- gpu-sr675 series: 32-34
- gpu-xd670: 30

**A100 partition:**
- gpu-sr670 series: 20-23 (tested separately with `-a100` suffix)

**Note:** These nodes are hardcoded in the `Snakefile`. To update the list, edit the `GPU_TESTS` variable in the Snakefile. You can run `./discover_nodes.sh` to see currently available nodes in each partition.

## How It Works

This workflow uses **Snakemake with the SLURM executor**, which means:

1. **You run Snakemake once** via `./run_gpu_test.sh`
2. **Snakemake analyzes** the workflow and determines which jobs to run
3. **Snakemake submits jobs to SLURM** automatically using `sbatch` behind the scenes
4. **Each GPU node test runs** as a separate SLURM job via `test_gpu_node.py`
5. **Snakemake waits** for all jobs to complete
6. **Snakemake runs** the summarization step once all tests finish

**You do NOT manually submit jobs with `sbatch`** - Snakemake handles all job submission, tracking, and coordination automatically.

## Output

Each node produces a JSON file with:
```json
{
  "node": "gpu-350-01",
  "hostname": "actual-hostname",
  "status": "SUCCESS" | "FAILED",
  "tests": {
    "environment": { ... },
    "pytorch": { ... },
    "cellpose": { ... },
    "model_loading": { ... },
    "inference": { ... }
  },
  "errors": [ ... ]
}
```

## Troubleshooting

### No results generated
- Check SLURM logs in `logs/slurm/`
- Verify conda environment is activated
- Check SLURM queue: `squeue -u $USER`

### All nodes fail
- Verify `pm` environment has correct packages installed
- Check PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Verify Cellpose is installed: `pip show cellpose`

### Some nodes fail
- Compare successful vs failed nodes in results
- Check if failures correlate with specific GPU types
- Examine error messages in individual JSON files

## Cleanup

To remove all results and start fresh:
```bash
rm -rf results/*.json results/summary.txt logs/slurm/*
```

To completely reset (including Snakemake metadata):
```bash
rm -rf results/*.json results/summary.txt logs/slurm/* .snakemake/
```
