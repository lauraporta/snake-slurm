#!/usr/bin/env python3
"""
Summarize GPU Test Results

Aggregates all GPU test results into a summary report.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Summarize GPU test results')
    parser.add_argument('--input-dir', required=True, help='Directory containing result JSON files')
    parser.add_argument('--output', required=True, help='Output summary text file')
    parser.add_argument('--input-files', nargs='+', help='List of input JSON files')
    args = parser.parse_args()
    
    # Load all results
    results = []
    if args.input_files:
        result_files = [Path(f) for f in args.input_files]
    else:
        result_files = sorted(Path(args.input_dir).glob("*_result.json"))
    
    for result_file in result_files:
        with open(result_file, 'r') as f:
            results.append(json.load(f))
    
    # Write summary
    with open(args.output, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CELLPOSE-SAM GPU TEST SUMMARY\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total nodes tested: {len(results)}\n")
        f.write("=" * 80 + "\n\n")
        
        # Count successes and failures
        successes = [r for r in results if r['status'] == 'SUCCESS']
        failures = [r for r in results if r['status'] == 'FAILED']
        
        f.write(f"✓ SUCCESS: {len(successes)} nodes\n")
        f.write(f"✗ FAILED:  {len(failures)} nodes\n\n")
        
        # List successful nodes
        if successes:
            f.write("SUCCESSFUL NODES:\n")
            f.write("-" * 80 + "\n")
            for r in successes:
                f.write(f"  ✓ {r['node']} (hostname: {r['hostname']})\n")
                if 'pytorch' in r['tests'] and 'devices' in r['tests']['pytorch']:
                    for dev in r['tests']['pytorch']['devices']:
                        f.write(f"      GPU: {dev['name']} ({dev['memory_gb']}GB)\n")
            f.write("\n")
        
        # List failed nodes with details
        if failures:
            f.write("FAILED NODES:\n")
            f.write("-" * 80 + "\n")
            for r in failures:
                f.write(f"  ✗ {r['node']} (hostname: {r['hostname']})\n")
                for error in r['errors']:
                    f.write(f"      ERROR: {error}\n")
                f.write("\n")
        
        # Detailed breakdown by error type
        f.write("\nDETAILED ERROR BREAKDOWN:\n")
        f.write("-" * 80 + "\n")
        
        error_categories = {}
        for r in failures:
            for error in r['errors']:
                if error not in error_categories:
                    error_categories[error] = []
                error_categories[error].append(r['node'])
        
        for error, nodes in error_categories.items():
            f.write(f"\n{error}\n")
            f.write(f"  Affected nodes ({len(nodes)}): {', '.join(nodes)}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"\nSummary written to: {args.output}")
    
    # Also print to console
    with open(args.output, 'r') as f:
        print(f.read())


if __name__ == "__main__":
    main()
