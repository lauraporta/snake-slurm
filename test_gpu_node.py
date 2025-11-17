#!/usr/bin/env python3
"""
GPU Node Testing Script for Cellpose-SAM

Tests a single GPU node for:
1. PyTorch CUDA availability
2. Cellpose-SAM model loading with GPU
3. Dummy inference to verify GPU execution
4. Logs all environment details for debugging

Outputs results to a JSON file.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import traceback


def test_environment():
    """Collect environment information."""
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "N/A"),
        "slurm_nodelist": os.environ.get("SLURM_NODELIST", "N/A"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "N/A"),
    }


def test_pytorch():
    """Test PyTorch and CUDA availability."""
    print("\n[TEST 1] PyTorch and CUDA availability")
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        
        result = {
            "version": torch.__version__,
            "cuda_compiled_version": torch.version.cuda,
            "cuda_available": cuda_available,
            "device_count": device_count,
        }
        
        if cuda_available:
            result["devices"] = []
            for i in range(device_count):
                device_info = {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2),
                    "compute_capability": f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}"
                }
                result["devices"].append(device_info)
                print(f"  ✓ GPU {i}: {device_info['name']} ({device_info['memory_gb']}GB)")
        else:
            print("  ✗ CUDA not available!")
            return result, ["CUDA not available despite GPU partition"]
            
        return result, []
        
    except Exception as e:
        print(f"  ✗ PyTorch test failed: {e}")
        return {"error": str(e), "traceback": traceback.format_exc()}, [f"PyTorch error: {e}"]


def test_cellpose():
    """Test Cellpose import and version."""
    print("\n[TEST 2] Cellpose import")
    try:
        import cellpose
        
        # Get version
        try:
            version = cellpose.__version__
        except AttributeError:
            result_proc = subprocess.run(
                ['pip', 'show', 'cellpose'], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            version = "unknown"
            for line in result_proc.stdout.split('\n'):
                if line.startswith('Version:'):
                    version = line.split(':', 1)[1].strip()
                    break
        
        result = {
            "version": version,
            "imported": True
        }
        print(f"  ✓ Cellpose version: {version}")
        
        return result, []
        
    except Exception as e:
        print(f"  ✗ Cellpose import failed: {e}")
        return {"error": str(e), "traceback": traceback.format_exc()}, [f"Cellpose import error: {e}"]


def test_model_loading():
    """Test Cellpose-SAM model loading with GPU."""
    print("\n[TEST 3] CellposeSAM model loading")
    try:
        import torch
        from cellpose import models
        
        print("  Loading CellposeModel with GPU=True, pretrained_model='cpsam'...")
        model = models.CellposeModel(gpu=True, pretrained_model='cpsam')
        
        result = {
            "success": True,
            "device": str(model.device),
            "gpu_flag": model.gpu,
        }
        
        # Check network device
        if hasattr(model, 'net') and model.net is not None:
            if hasattr(model.net, 'device'):
                result["network_device"] = str(model.net.device)
            
            # Check if parameters are on GPU
            try:
                first_param = next(model.net.parameters())
                result["parameter_device"] = str(first_param.device)
                print(f"  ✓ Model loaded on: {first_param.device}")
            except:
                pass
        
        # Warning if on CPU despite GPU request
        errors = []
        if model.device == 'cpu' and torch.cuda.is_available():
            warning = "Model loaded on CPU despite CUDA being available!"
            print(f"  ⚠️  {warning}")
            errors.append(warning)
            result["warning"] = warning
        else:
            print(f"  ✓ Model successfully loaded on: {model.device}")
        
        return model, result, errors
            
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        return None, {"error": str(e), "traceback": traceback.format_exc()}, [f"Model loading error: {e}"]


def test_inference(model):
    """Test dummy inference."""
    print("\n[TEST 4] Dummy inference")
    try:
        import numpy as np
        
        # Create small test image
        dummy_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        print("  Running inference on 256x256 dummy image...")
        
        masks, flows, styles = model.eval(dummy_img, diameter=None, channels=[0, 0])
        
        num_masks = len(np.unique(masks)) - 1  # Subtract background
        result = {
            "success": True,
            "num_masks_found": num_masks,
            "output_shape": str(masks.shape)
        }
        
        print(f"  ✓ Inference successful")
        print(f"  ✓ Found {num_masks} masks")
        
        return result, []
        
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        return {"error": str(e), "traceback": traceback.format_exc()}, [f"Inference error: {e}"]


def main():
    parser = argparse.ArgumentParser(description='Test GPU node for Cellpose-SAM')
    parser.add_argument('--node', required=True, help='Node name being tested')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    args = parser.parse_args()
    
    result = {
        "node": args.node,
        "hostname": socket.gethostname(),
        "status": "unknown",
        "tests": {},
        "errors": []
    }
    
    print("=" * 60)
    print(f"TESTING GPU NODE: {args.node}")
    print(f"Actual hostname: {socket.gethostname()}")
    print("=" * 60)
    
    # Test 1: Environment info
    result["tests"]["environment"] = test_environment()
    
    # Test 2: PyTorch and CUDA
    pytorch_result, pytorch_errors = test_pytorch()
    result["tests"]["pytorch"] = pytorch_result
    result["errors"].extend(pytorch_errors)
    
    # Test 3: Cellpose import
    cellpose_result, cellpose_errors = test_cellpose()
    result["tests"]["cellpose"] = cellpose_result
    result["errors"].extend(cellpose_errors)
    
    # Test 4: Model loading
    model, model_result, model_errors = test_model_loading()
    result["tests"]["model_loading"] = model_result
    result["errors"].extend(model_errors)
    
    # Test 5: Inference (only if model loaded successfully)
    if model is not None:
        inference_result, inference_errors = test_inference(model)
        result["tests"]["inference"] = inference_result
        result["errors"].extend(inference_errors)
    else:
        result["tests"]["inference"] = {"error": "Model not loaded, skipping inference"}
    
    # Determine overall status
    if len(result["errors"]) == 0:
        result["status"] = "SUCCESS"
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
    else:
        result["status"] = "FAILED"
        print("\n" + "=" * 60)
        print(f"TESTS FAILED: {len(result['errors'])} error(s)")
        print("=" * 60)
    
    # Save result to JSON
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    
    # Exit with error code if tests failed
    sys.exit(0 if result["status"] == "SUCCESS" else 1)


if __name__ == "__main__":
    main()
