"""
Analysis of consistency between format pairs that differ by only one formatting element.

This script analyzes the effect of individual formatting elements on consistency by 
measuring pairwise consistency between formats that differ by exactly one bit.

Format encoding (3-bit binary):
- Format 0: 000, Format 1: 001, Format 2: 010, Format 3: 011
- Format 4: 100, Format 5: 101, Format 6: 110, Format 7: 111

Analysis pairs:
- Last bit effect: (000,001), (010,011), (100,101), (110,111)
- Middle bit effect: (000,010), (001,011), (100,110), (101,111)  
- First bit effect: (000,100), (001,101), (010,110), (011,111)
"""

import os
import json
import jsonlines
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict

def get_format_pairs_by_position() -> Dict[str, List[Tuple[int, int]]]:
    """
    Get format pairs that differ by exactly one bit at each position.
    
    Returns:
        Dictionary mapping position names to list of format pairs
    """
    # All 8 formats in 3-bit binary representation
    formats = list(range(8))
    
    pairs = {
        "last_bit": [],      # bit 0 (rightmost)
        "middle_bit": [],    # bit 1 (middle)  
        "first_bit": []      # bit 2 (leftmost)
    }
    
    # Find pairs that differ by exactly one bit
    for i in formats:
        for j in formats:
            if i < j:  # Only consider unique pairs
                # Convert to binary and compare
                i_bits = f"{i:03b}"
                j_bits = f"{j:03b}"
                
                # Count differences
                diff_positions = [k for k in range(3) if i_bits[k] != j_bits[k]]
                
                # If exactly one bit differs, categorize by position
                if len(diff_positions) == 1:
                    pos = diff_positions[0]
                    if pos == 0:  # First bit (leftmost in binary string)
                        pairs["first_bit"].append((i, j))
                    elif pos == 1:  # Middle bit
                        pairs["middle_bit"].append((i, j))
                    elif pos == 2:  # Last bit (rightmost in binary string)
                        pairs["last_bit"].append((i, j))
    
    return pairs

def calculate_pairwise_consistency(predictions_1: List[str], predictions_2: List[str]) -> float:
    """
    Calculate consistency between two sets of predictions.
    
    Args:
        predictions_1: List of predictions for format 1
        predictions_2: List of predictions for format 2
    
    Returns:
        Consistency score (fraction of matching predictions)
    """
    if len(predictions_1) != len(predictions_2):
        return 0.0
    
    matches = 0
    valid_pairs = 0
    
    for p1, p2 in zip(predictions_1, predictions_2):
        # Skip None or invalid predictions
        if p1 is not None and p2 is not None and p1 != "None" and p2 != "None":
            valid_pairs += 1
            if p1 == p2:
                matches += 1
    
    return matches / valid_pairs if valid_pairs > 0 else 0.0

def analyze_single_element_consistency(data_path: str) -> Dict[str, Dict[str, float]]:
    """
    Analyze consistency for format pairs differing by single elements.
    
    Args:
        data_path: Path to the predictions JSONL file
        
    Returns:
        Dictionary with consistency scores for each position and pair
    """
    pairs_by_position = get_format_pairs_by_position()
    results = {
        "last_bit": {},
        "middle_bit": {}, 
        "first_bit": {}
    }
    
    # Read data
    with jsonlines.open(data_path) as fin:
        data = list(fin)
    
    # For each position (first, middle, last bit)
    for position, pairs in pairs_by_position.items():
        position_consistencies = []
        
        # For each pair in this position
        for format_1, format_2 in pairs:
            pair_consistencies = []
            
            # For each example in the dataset
            for example in data:
                if str(format_1) in example["predictions"] and str(format_2) in example["predictions"]:
                    pred_1 = example["predictions"][str(format_1)]
                    pred_2 = example["predictions"][str(format_2)]
                    
                    # Calculate consistency for this example
                    if pred_1 is not None and pred_2 is not None and pred_1 != "None" and pred_2 != "None":
                        consistency = 1.0 if pred_1 == pred_2 else 0.0
                        pair_consistencies.append(consistency)
            
            # Calculate mean consistency for this pair
            if pair_consistencies:
                mean_consistency = np.mean(pair_consistencies)
                results[position][f"({format_1:03b},{format_2:03b})"] = mean_consistency
                position_consistencies.append(mean_consistency)
        
        # Calculate overall mean for this position
        if position_consistencies:
            results[position]["mean"] = np.mean(position_consistencies)
        else:
            results[position]["mean"] = 0.0
    
    return results

def analyze_dataset_model_combination(
    dataset_name: str, 
    model_name: str, 
    prompting_strategy: str,
    base_path: str = "../../../results_shared"
) -> Dict[str, Dict[str, float]]:
    """
    Analyze single element consistency for a specific dataset-model-strategy combination.
    
    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model
        prompting_strategy: Prompting strategy used
        base_path: Base path to results directory
        
    Returns:
        Dictionary with consistency analysis results
    """
    data_path = os.path.join(base_path, dataset_name, model_name, f"{prompting_strategy}_predictions.jsonl")
    
    if not os.path.exists(data_path):
        print(f"Warning: File not found: {data_path}")
        return None
    
    try:
        results = analyze_single_element_consistency(data_path)
        return results
    except Exception as e:
        print(f"Error analyzing {dataset_name}/{model_name}/{prompting_strategy}: {e}")
        return None

def generate_analysis_report(
    dataset_names: List[str],
    model_names: List[str], 
    prompting_strategies: List[str],
    output_dir: str = "."
) -> None:
    """
    Generate comprehensive analysis report for all combinations.
    
    Args:
        dataset_names: List of dataset names
        model_names: List of model names  
        prompting_strategies: List of prompting strategies
        output_dir: Output directory for results
    """
    all_results = {}
    
    for dataset_name in dataset_names:
        all_results[dataset_name] = {}
        
        for model_name in model_names:
            all_results[dataset_name][model_name] = {}
            
            for prompting_strategy in prompting_strategies:
                print(f"Analyzing {dataset_name}/{model_name}/{prompting_strategy}...")
                
                results = analyze_dataset_model_combination(
                    dataset_name, model_name, prompting_strategy
                )
                
                if results is not None:
                    all_results[dataset_name][model_name][prompting_strategy] = results
                else:
                    all_results[dataset_name][model_name][prompting_strategy] = None
    
    # Save detailed results
    # output_path = os.path.join(output_dir, "single_element_consistency_detailed.json")
    # with open(output_path, "w") as fout:
    #     json.dump(all_results, fout, indent=2)
    
    # Generate summary statistics
    generate_summary_statistics(all_results, output_dir)
    
    # Print analysis coverage table
    print_analysis_coverage(all_results, dataset_names, model_names, prompting_strategies)
    
    print(f"Analysis complete.")

def generate_summary_statistics(all_results: Dict, output_dir: str) -> None:
    """
    Generate summary statistics and tables from detailed results.
    
    Args:
        all_results: Dictionary containing all analysis results
        output_dir: Output directory for summary files
    """
    # Create summary tables for each bit position
    positions = ["first_bit", "middle_bit", "last_bit"]
    
    for position in positions:
        summary_data = []
        
        for dataset_name, dataset_results in all_results.items():
            for model_name, model_results in dataset_results.items():
                for strategy_name, strategy_results in model_results.items():
                    if strategy_results is not None and position in strategy_results:
                        row = {
                            "Dataset": dataset_name,
                            "Model": model_name, 
                            "Strategy": strategy_name,
                            "Mean_Consistency": strategy_results[position].get("mean", 0.0)
                        }
                        
                        # Add individual pair consistencies
                        for pair, consistency in strategy_results[position].items():
                            if pair != "mean":
                                row[f"Pair_{pair}"] = consistency
                        
                        summary_data.append(row)
        
        # Create DataFrame and save
        if summary_data:
            df = pd.DataFrame(summary_data)
            # output_path = os.path.join(output_dir, f"{position}_consistency_summary.csv")
            # df.to_csv(output_path, index=False)
            
            # Print summary statistics
            print(f"\n{position.replace('_', ' ').title()} Effect Summary:")
            print(f"Mean consistency across all combinations: {df['Mean_Consistency'].mean():.3f}")
            print(f"Standard deviation: {df['Mean_Consistency'].std():.3f}")
            # print(f"Min consistency: {df['Mean_Consistency'].min():.3f}")
            # print(f"Max consistency: {df['Mean_Consistency'].max():.3f}")
            
            # Find and print min/max cases
            min_idx = df['Mean_Consistency'].idxmin()
            max_idx = df['Mean_Consistency'].idxmax()
            
            min_case = df.loc[min_idx]
            max_case = df.loc[max_idx]
            
            print(f"Min case = {min_case['Mean_Consistency']:.3f} ( {min_case['Dataset']} / {min_case['Model']} / {min_case['Strategy']} )")
            print(f"Max case = {max_case['Mean_Consistency']:.3f} ( {max_case['Dataset']} / {max_case['Model']} / {max_case['Strategy']} )")
            # print(f"Max case: {max_case['Dataset']}/{max_case['Model']}/{max_case['Strategy']} = {max_case['Mean_Consistency']:.3f}")

def print_analysis_coverage(
    all_results: Dict, 
    dataset_names: List[str], 
    model_names: List[str], 
    prompting_strategies: List[str]
) -> None:
    """
    Print a table showing which dataset/model/strategy combinations were analyzed.
    
    Args:
        all_results: Dictionary containing all analysis results
        dataset_names: List of dataset names
        model_names: List of model names  
        prompting_strategies: List of prompting strategies
    """
    print("\n" + "="*120)
    print("ANALYSIS COVERAGE TABLE")
    print("="*120)
    
    # Count successful and failed analyses
    total_combinations = len(dataset_names) * len(model_names) * len(prompting_strategies)
    successful_count = 0
    failed_count = 0
    
    # Helper function to get status
    def get_status(dataset, model, strategy):
        if (dataset in all_results and 
            model in all_results[dataset] and 
            strategy in all_results[dataset][model] and
            all_results[dataset][model][strategy] is not None):
            return "O"
        else:
            return "✗"
    
    # Print table header
    print(f"\n{'Dataset':<15} {'Model':<35} {'zero-shot':<12} {'zero-shot-cot':<15} {'few-shot':<12} {'few-shot-cot':<13}")
    print("-" * 120)
    
    # Print table rows
    for dataset_name in dataset_names:
        for model_name in model_names:
            # Truncate long model names for better formatting
            display_model = model_name if len(model_name) <= 33 else model_name[:30] + "..."
            
            # Get status for each strategy
            status_zero_shot = get_status(dataset_name, model_name, "zero-shot")
            status_zero_shot_cot = get_status(dataset_name, model_name, "zero-shot-cot")
            status_few_shot = get_status(dataset_name, model_name, "few-shot")
            status_few_shot_cot = get_status(dataset_name, model_name, "few-shot-cot")
            
            # Count successes for overall stats
            for status in [status_zero_shot, status_zero_shot_cot, status_few_shot, status_few_shot_cot]:
                if status == "O":
                    successful_count += 1
                else:
                    failed_count += 1
            
            print(f"{dataset_name:<15} {display_model:<35} {status_zero_shot:<12} {status_zero_shot_cot:<15} {status_few_shot:<12} {status_few_shot_cot:<13}")
    
    print("-" * 120)
    print(f"Total combinations: {total_combinations}")
    print(f"Successful analyses: {successful_count}")
    print(f"Failed analyses: {failed_count}")
    print(f"Success rate: {successful_count/total_combinations*100:.1f}%")
    
    # Print summary by category
    print(f"\nSUMMARY BY CATEGORY:")
    print("-" * 40)
    
    # By dataset
    print("\nBy Dataset:")
    for dataset_name in dataset_names:
        dataset_success = 0
        dataset_total = 0
        for model_name in model_names:
            for strategy_name in prompting_strategies:
                dataset_total += 1
                if get_status(dataset_name, model_name, strategy_name) == "O":
                    dataset_success += 1
        
        success_rate = dataset_success/dataset_total*100 if dataset_total > 0 else 0
        print(f"  {dataset_name:<15}: {dataset_success:>2}/{dataset_total:<2} ({success_rate:>5.1f}%)")
    
    # By model
    print("\nBy Model:")
    for model_name in model_names:
        model_success = 0
        model_total = 0
        for dataset_name in dataset_names:
            for strategy_name in prompting_strategies:
                model_total += 1
                if get_status(dataset_name, model_name, strategy_name) == "O":
                    model_success += 1
        
        success_rate = model_success/model_total*100 if model_total > 0 else 0
        display_model = model_name if len(model_name) <= 28 else model_name[:25] + "..."
        print(f"  {display_model:<30}: {model_success:>2}/{model_total:<2} ({success_rate:>5.1f}%)")
    
    # By strategy
    print("\nBy Strategy:")
    for strategy_name in prompting_strategies:
        strategy_success = 0
        strategy_total = 0
        for dataset_name in dataset_names:
            for model_name in model_names:
                strategy_total += 1
                if get_status(dataset_name, model_name, strategy_name) == "O":
                    strategy_success += 1
        
        success_rate = strategy_success/strategy_total*100 if strategy_total > 0 else 0
        print(f"  {strategy_name:<15}: {strategy_success:>2}/{strategy_total:<2} ({success_rate:>5.1f}%)")
    
    print("="*120)

def main():
    """Main function to run the analysis."""
    # Define datasets and models to analyze
    dataset_names = [
        "CommonsenseQA", 
        "QASC",
        "100TFQA", 
        "GSM8K", 
        ]
    model_names = [
        "Phi-3.5-mini-instruct",
        "Phi-3.5-vision-instruct",
        "Llama-3.1-8B",
        "Llama-3.1-8B-Instruct", 
        "DeepSeek-R1-Distill-Llama-8B", 
        "gpt-4o-2024-11-20",
    ]
    prompting_strategies = ["zero-shot", "zero-shot-cot", "few-shot", "few-shot-cot"]
    
    # Create output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Print format pair information
    print("\nFormat pair analysis:")
    pairs_by_position = get_format_pairs_by_position()
    for position, pairs in pairs_by_position.items():
        print(f"\n{position.replace('_', ' ').title()} pairs:")
        for format_1, format_2 in pairs:
            print(f"  ({format_1:03b}, {format_2:03b}) = (Format {format_1}, Format {format_2})")

    # Run analysis
    generate_analysis_report(dataset_names, model_names, prompting_strategies, output_dir)
    
    

if __name__ == "__main__":
    main()
