"""
Snakemake workflow to test Cellpose-SAM across all GPU nodes.

This workflow:
- Submits jobs via Snakemake's SLURM executor (not direct sbatch)
- Tests each GPU node for PyTorch CUDA, model loading, and inference
- Aggregates results into a summary report

Each job calls test_gpu_node.py which performs the actual testing.
"""

from pathlib import Path

# Define GPU nodes to test organized by partition
# Each entry is (node_name, partition)
# Discovered via: sinfo -p <partition> -o "%N" -h
GPU_TESTS = [
    # ============================================
    # GPU PARTITION (standard GPU nodes)
    # ============================================
    # gpu-350 series (gpu partition)
    ("gpu-350-01", "gpu"), ("gpu-350-02", "gpu"), ("gpu-350-03", "gpu"), 
    ("gpu-350-04", "gpu"), ("gpu-350-05", "gpu"),
    
    # gpu-380 series (gpu partition)
    ("gpu-380-10", "gpu"), ("gpu-380-11", "gpu"), ("gpu-380-12", "gpu"), 
    ("gpu-380-13", "gpu"), ("gpu-380-14", "gpu"), ("gpu-380-15", "gpu"), 
    ("gpu-380-16", "gpu"), ("gpu-380-17", "gpu"), ("gpu-380-18", "gpu"),
    
    # gpu-sr670 series (gpu partition)
    ("gpu-sr670-20", "gpu"), ("gpu-sr670-21", "gpu"), 
    ("gpu-sr670-22", "gpu"), ("gpu-sr670-23", "gpu"),
    
    # gpu-sr675-31 (gpu partition only)
    ("gpu-sr675-31", "gpu"),
    
    # ============================================
    # GPU_LOWP PARTITION (low priority GPU nodes)
    # ============================================
    # gpu-sr675 series (gpu_lowp partition)
    ("gpu-sr675-32", "gpu_lowp"), ("gpu-sr675-33", "gpu_lowp"), 
    ("gpu-sr675-34", "gpu_lowp"),
    
    # gpu-xd670 series (gpu_lowp partition)
    ("gpu-xd670-30", "gpu_lowp"),
    
    # ============================================
    # A100 PARTITION (A100 GPU nodes)
    # ============================================
    # gpu-sr670 series also available in a100 partition
    # Testing same nodes in a100 partition with suffix to distinguish
    ("gpu-sr670-20-a100", "a100"), ("gpu-sr670-21-a100", "a100"), 
    ("gpu-sr670-22-a100", "a100"), ("gpu-sr670-23-a100", "a100"),
]

# Extract unique node names for wildcard constraints
GPU_NODES = [node for node, partition in GPU_TESTS]

# Create a mapping from node to partition for easy lookup
NODE_TO_PARTITION = {node: partition for node, partition in GPU_TESTS}

# Configuration from config file or defaults
slurm_config = config.get("slurm", {})


rule all:
    input:
        expand("results/{node}_result.json", node=GPU_NODES),
        "results/summary.txt"


rule test_gpu_node:
    """Test Cellpose-SAM model loading and inference on a specific GPU node."""
    output:
        result="results/{node}_result.json"
    params:
        node="{node}",
        partition=lambda wildcards: NODE_TO_PARTITION[wildcards.node],
        # Extract actual node name (strip -a100 suffix if present)
        actual_node=lambda wildcards: wildcards.node.replace("-a100", "")
    resources:
        slurm_partition=lambda wildcards: NODE_TO_PARTITION[wildcards.node],
        mem_mb=slurm_config.get("mem_mb", 8000),
        runtime=slurm_config.get("runtime", 10),
        tasks=1,
        nodes=1,
        gpu=slurm_config.get("gpu", 1),
        tasks_per_gpu=slurm_config.get("tasks_per_gpu", 0),
        slurm_extra=lambda wildcards: f"--nodelist={wildcards.node.replace('-a100', '')}"
    shell:
        """
        python test_gpu_node.py --node {params.node} --output {output.result}
        """


rule summarize_results:
    """Aggregate all GPU test results into a summary report."""
    input:
        expand("results/{node}_result.json", node=GPU_NODES)
    output:
        summary="results/summary.txt"
    shell:
        """
        python summarize_results.py --input-files {input} --output {output.summary}
        """
