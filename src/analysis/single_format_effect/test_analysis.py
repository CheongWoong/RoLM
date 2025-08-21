#!/usr/bin/env python3
"""
Quick test script for single format element consistency analysis.

This script provides a simple way to test the analysis on a specific dataset/model combination
and generate quick results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from single_format_element_consistency import (
    get_format_pairs_by_position,
    analyze_dataset_model_combination
)

def test_analysis():
    """Test the analysis on a sample combination."""
    print("Testing single format element consistency analysis...")
    print("=" * 60)
    
    # Show format pairs
    pairs_by_position = get_format_pairs_by_position()
    print("Format pairs that differ by exactly one bit:")
    print()
    
    for position, pairs in pairs_by_position.items():
        print(f"{position.replace('_', ' ').title()} effect pairs:")
        for format_1, format_2 in pairs:
            print(f"  ({format_1:03b}, {format_2:03b}) = (Format {format_1}, Format {format_2})")
        print()
    
    # Test on sample data
    print("Testing on sample data: 100TFQA/DeepSeek-R1-Distill-Llama-8B/few-shot-cot")
    print("-" * 60)
    
    results = analyze_dataset_model_combination(
        dataset_name="100TFQA",
        model_name="DeepSeek-R1-Distill-Llama-8B",
        prompting_strategy="few-shot-cot"
    )
    
    if results:
        print("Analysis Results:")
        print()
        
        for position, position_results in results.items():
            print(f"{position.replace('_', ' ').title()} Effect:")
            print(f"  Overall mean consistency: {position_results.get('mean', 0.0):.3f}")
            
            for pair, consistency in position_results.items():
                if pair != "mean":
                    print(f"  Pair {pair}: {consistency:.3f}")
            print()
            
        # Calculate and display relative effects
        means = [results[pos].get('mean', 0.0) for pos in ['first_bit', 'middle_bit', 'last_bit']]
        
        print("Relative Effects:")
        position_names = ['First bit', 'Middle bit', 'Last bit']
        for i, (name, mean_val) in enumerate(zip(position_names, means)):
            print(f"  {name}: {mean_val:.3f}")
        
        if all(m > 0 for m in means):
            max_pos = position_names[means.index(max(means))]
            min_pos = position_names[means.index(min(means))]
            print(f"\n\n  Most consistent position: {max_pos} ({max(means):.3f})")
            print(f"  Least consistent position: {min_pos} ({min(means):.3f})")
            print(f"  Difference: {max(means) - min(means):.3f}")
    else:
        print("Could not load data. Please check that the file exists:")
        print("../../../results_shared/100TFQA/DeepSeek-R1-Distill-Llama-8B/few-shot-cot_predictions.jsonl")

def run_quick_comparison():
    """Run a quick comparison across different strategies for one model."""
    print("\n\n" + "=" * 60)
    print("Quick comparison across prompting strategies")
    print("Model: DeepSeek-R1-Distill-Llama-8B, Dataset: 100TFQA")
    print("-" * 60)
    
    strategies = ["zero-shot", "zero-shot-cot", "few-shot", "few-shot-cot"]
    
    for strategy in strategies:
        print(f"\n\nStrategy: {strategy}")
        
        results = analyze_dataset_model_combination(
            dataset_name="100TFQA",
            model_name="DeepSeek-R1-Distill-Llama-8B",
            prompting_strategy=strategy
        )
        
        if results:
            for position in ['first_bit', 'middle_bit', 'last_bit']:
                mean_consistency = results[position].get('mean', 0.0)
                print(f"  {position.replace('_', ' ').title()}: {mean_consistency:.3f}")
        else:
            print(f"  No data available for {strategy}")

if __name__ == "__main__":
    test_analysis()
    run_quick_comparison()
