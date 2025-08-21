#!/usr/bin/env python3
"""
Simple script to show average format pair consistency.
"""

import sys
import os
import numpy as np
import jsonlines
from collections import defaultdict

sys.path.append('.')

def calculate_format_pair_consistency(data, format1, format2):
    """Calculate consistency between two specific formats."""
    matches = 0
    valid_pairs = 0
    
    for example in data:
        if str(format1) in example["predictions"] and str(format2) in example["predictions"]:
            pred1 = example["predictions"][str(format1)]
            pred2 = example["predictions"][str(format2)]
            
            if pred1 is not None and pred1 != "None" and pred2 is not None and pred2 != "None":
                valid_pairs += 1
                if pred1 == pred2:
                    matches += 1
    
    return matches / valid_pairs if valid_pairs > 0 else 0.0

def get_format_description(format_num):
    """Get human-readable description of format combination."""
    first_bit = (format_num >> 2) & 1   # Separator: 0=':' 1=': '
    middle_bit = (format_num >> 1) & 1  # Casing: 0='Question' 1='QUESTION'  
    last_bit = format_num & 1           # Space: 0='Question' 1=' Question'
    
    separator = ': ' if first_bit else ':'
    casing = 'UPPER' if middle_bit else 'Title'
    space = 'spaced' if last_bit else 'nospace'
    
    return f"{separator}|{casing}|{space}"

def get_bit_difference_description(f1, f2):
    """Get description of which bit differs between two formats."""
    xor = f1 ^ f2
    
    if xor == 4:  # 100 - first bit differs
        return "Separator"
    elif xor == 2:  # 010 - middle bit differs  
        return "Casing"
    elif xor == 1:  # 001 - last bit differs
        return "Spacing"
    else:
        # Multiple bits differ
        bits = []
        if xor & 4:
            bits.append("Separator")
        if xor & 2:
            bits.append("Casing")
        if xor & 1:
            bits.append("Spacing")
        return "+".join(bits)

def main():
    print("Format Combination Encoding (3-bit binary):")
    for i in range(8):
        binary = f"{i:03b}"
        desc = get_format_description(i)
        print(f"  Format {i}: {binary} - {desc}")
    
    # Find all existing combinations
    dataset_names = ["CommonsenseQA", "QASC", "100TFQA", "GSM8K"]
    model_names = [
        "Phi-3.5-mini-instruct",
        "Phi-3.5-vision-instruct", 
        "Llama-3.1-8B",
        "Llama-3.1-8B-Instruct",
        "DeepSeek-R1-Distill-Llama-8B",
        "gpt-4o-2024-11-20"
    ]
    prompting_strategies = ["zero-shot", "zero-shot-cot", "few-shot", "few-shot-cot"]
    
    all_combinations = []
    base_path = "../../../results"
    
    for dataset in dataset_names:
        for model in model_names:
            for strategy in prompting_strategies:
                data_path = os.path.join(base_path, dataset, model, f"{strategy}_predictions.jsonl")
                if os.path.exists(data_path):
                    all_combinations.append((dataset, model, strategy))
    
    print(f"\nFound {len(all_combinations)} valid dataset/model/strategy combinations")
    
    # Collect all pairwise consistencies across all settings
    pair_consistencies = defaultdict(list)
    
    for dataset, model, strategy in all_combinations:
        data_path = os.path.join(base_path, dataset, model, f"{strategy}_predictions.jsonl")
        
        with jsonlines.open(data_path) as fin:
            data = list(fin)
        
        # Calculate consistency for all format pairs
        for i in range(8):
            for j in range(i+1, 8):
                consistency = calculate_format_pair_consistency(data, i, j)
                pair_key = f"({i:03b},{j:03b})"
                pair_consistencies[pair_key].append(consistency)
    
    # Calculate averages and sort by consistency
    pair_averages = []
    for pair_key, consistencies in pair_consistencies.items():
        if consistencies:
            avg_consistency = np.mean(consistencies)
            std_consistency = np.std(consistencies)
            pair_averages.append((pair_key, avg_consistency, std_consistency, len(consistencies)))
    
    # Sort by average consistency
    pair_averages.sort(key=lambda x: x[1])
    
    print(f"\n{'='*80}")
    print(f"AVERAGE FORMAT PAIR CONSISTENCY ANALYSIS")
    print(f"{'='*80}")
    
    print(f"\nAverage Consistency by Format Pair (sorted by consistency):")
    print(f"{'Pair':<12} {'Formats':<15} {'AvgConsist':<12} {'StdDev':<10} {'Count':<8} {'Description'}")
    print("-" * 85)
    
    for pair_key, avg_cons, std_cons, count in pair_averages:
        # Extract format numbers
        f1_str, f2_str = pair_key.strip("()").split(",")
        f1 = int(f1_str, 2)
        f2 = int(f2_str, 2)
        
        # Get bit difference description
        diff_desc = get_bit_difference_description(f1, f2)
        format_desc = f"F{f1}↔F{f2}"
        
        print(f"{pair_key:<12} {format_desc:<15} {avg_cons:<12.3f} {std_cons:<10.3f} {count:<8} {diff_desc}")
    
    # Analysis by bit position
    print(f"\nAnalysis by Format Element (Bit Position):")
    
    # Group pairs by bit difference
    first_bit_pairs = []   # Separator differences
    middle_bit_pairs = []  # Casing differences
    last_bit_pairs = []    # Space differences
    
    for pair_key, avg_cons, std_cons, count in pair_averages:
        f1_str, f2_str = pair_key.strip("()").split(",")
        f1 = int(f1_str, 2)
        f2 = int(f2_str, 2)
        
        # Determine which bit differs
        xor = f1 ^ f2
        if xor == 4:  # 100 - first bit differs
            first_bit_pairs.append((pair_key, avg_cons, std_cons))
        elif xor == 2:  # 010 - middle bit differs
            middle_bit_pairs.append((pair_key, avg_cons, std_cons))
        elif xor == 1:  # 001 - last bit differs
            last_bit_pairs.append((pair_key, avg_cons, std_cons))
    
    # Calculate and display averages by bit position
    def print_bit_analysis(pairs, bit_name, description):
        if pairs:
            avg_values = [cons for _, cons, _ in pairs]
            overall_avg = np.mean(avg_values)
            overall_std = np.std(avg_values)
            
            print(f"\n{bit_name} Effect ({description}):")
            print(f"  Overall average: {overall_avg:.3f} ± {overall_std:.3f}")
            print(f"  Individual pairs:")
            
            for pair_key, avg_cons, std_cons in sorted(pairs, key=lambda x: x[1]):
                print(f"    {pair_key}: {avg_cons:.3f} ± {std_cons:.3f}")
    
    print_bit_analysis(first_bit_pairs, "Separator", "':' vs ': '")
    print_bit_analysis(middle_bit_pairs, "Casing", "'Question' vs 'QUESTION'")
    print_bit_analysis(last_bit_pairs, "Spacing", "'Question' vs ' Question'")
    
    # Summary statistics
    all_consistencies = [avg for _, avg, _, _ in pair_averages]
    
    print(f"\nOverall Summary:")
    print(f"  Mean consistency across all format pairs: {np.mean(all_consistencies):.3f}")
    print(f"  Standard deviation: {np.std(all_consistencies):.3f}")
    print(f"  Range: {np.min(all_consistencies):.3f} to {np.max(all_consistencies):.3f}")
    
    # Most and least consistent pairs
    print(f"\nMost Consistent Format Pairs:")
    for i, (pair_key, avg_cons, std_cons, count) in enumerate(pair_averages[-5:], 1):
        f1_str, f2_str = pair_key.strip("()").split(",")
        f1 = int(f1_str, 2)
        f2 = int(f2_str, 2)
        diff_desc = get_bit_difference_description(f1, f2)
        print(f"  {i}. {pair_key}: {avg_cons:.3f} ± {std_cons:.3f} ({diff_desc})")
    
    print(f"\nLeast Consistent Format Pairs:")
    for i, (pair_key, avg_cons, std_cons, count) in enumerate(pair_averages[:5], 1):
        f1_str, f2_str = pair_key.strip("()").split(",")
        f1 = int(f1_str, 2)
        f2 = int(f2_str, 2)
        diff_desc = get_bit_difference_description(f1, f2)
        print(f"  {i}. {pair_key}: {avg_cons:.3f} ± {std_cons:.3f} ({diff_desc})")

if __name__ == "__main__":
    main()
