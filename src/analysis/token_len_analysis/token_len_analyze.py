import os
import json
import jsonlines
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import pointbiserialr, pearsonr
from sklearn.linear_model import LogisticRegression
import numpy as np
from utils.counter import analyze_token_length
from collections import defaultdict

def micro_analysis(tokenlen_path, answers_path, formats=range(8)):
    # 1) Read token lengths
    # tokenlen.jsonl structure:
    # {"id": "...", "token_len": {"0": 132, "1": 133, ... }, "diff_token_len": {"0": 0, "1": 1, ...}, "min_token_len": 132, "min_format": ["0"]}
    tokenlen_records = {}
    min_format = {}
    with jsonlines.open(tokenlen_path) as fin:
        for record in fin:
            sample_id = record['id']
            tokenlen_records[sample_id] = record['token_len']
            min_format[sample_id] = record['min_format']

    # 2) Read answers
    # predictions.jsonl structure:
    # { "id": ..., "answers": {"0": "some text", "1": "another text", ...}, ... }
    answer_records = {}
    with jsonlines.open(answers_path) as fin:
        for record in fin:
            sample_id = record['id']
            answer_records[sample_id] = record['predictions']

    rows = []
    same_len_consistency_list = defaultdict(list)
    diff_len_consistency_list = defaultdict(list)
    min_len_consistency_list = []
    ran_consitency_list = []
    for sample_id in tokenlen_records:
        if sample_id not in answer_records:
            raise ValueError(f"Sample ID {sample_id} not found in answers")
            continue

        token_lens = tokenlen_records[sample_id]
        answers = answer_records[sample_id]

        for f1, f2 in combinations(formats, 2):
            f1_str, f2_str = str(f1), str(f2)
            if f1_str not in token_lens or f2_str not in token_lens:
                raise ValueError(f"Format {f1_str} or {f2_str} not found in token_lens")
                continue
            if f1_str not in answers or f2_str not in answers:
                raise ValueError(f"Format {f1_str} or {f2_str} not found in answers")
                continue

            diff = abs(token_lens[f1_str] - token_lens[f2_str])

            same_answer = int(answers[f1_str] == answers[f2_str])
            if diff == 0:
                same_len_consistency_list["format"+f1_str + f2_str].append(same_answer)
            else:
                diff_len_consistency_list["format"+f1_str + f2_str].append(same_answer)

            rows.append({
                'sample_id': sample_id,
                'f1': f1_str,
                'f2': f2_str,
                'token_diff': diff,
                'same_answer': same_answer
            })

        min_len_consistency_list.append(1 if len(set([answers[f] for f in min_format[sample_id]])) == 1 else 0)
        
        from random import choice
        diff_formats = [str(f) for f in formats]
        for min_f in min_format[sample_id]:
            diff_formats.remove(min_f)
        min_answers = list(set([answers[f] for f in min_format[sample_id]]))
        if len(min_answers) == 1:
            ran_consitency_list.append(1 if  min_answers[0] == answers[choice(diff_formats)] else 0)

    df = pd.DataFrame(rows)
    consistency = (same_len_consistency_list, diff_len_consistency_list, min_len_consistency_list, ran_consitency_list)
    return df, consistency


def run_micro_analysis(tokenlen_path, answers_path, output_path):
    df, consistency = micro_analysis(
        tokenlen_path=tokenlen_path,
        answers_path=answers_path
    )
    same_len_consistency_list, diff_len_consistency_list,min_len_consistency_list, ran_consistency_list = consistency

    df['token_diff_binary'] = (df['token_diff'] > 0).astype(int)

    # print(df.head())

    # A) token diff -> consistency ratio
    from pprint import pprint
    same_consistency_ratio = {format_pair: (sum(same_len_consistency_list[format_pair])/len(same_len_consistency_list[format_pair]), len(same_len_consistency_list[format_pair])) for format_pair in same_len_consistency_list}
    same_consistency_ratio = {format_pair: (round(same_consistency_ratio[format_pair][0], 5), same_consistency_ratio[format_pair][1]) for format_pair in same_consistency_ratio}
    # print("Token same -> consistency ratio:")

    diff_consistency_ratio = {format_pair: (sum(diff_len_consistency_list[format_pair])/len(diff_len_consistency_list[format_pair]), len(diff_len_consistency_list[format_pair])) for format_pair in diff_len_consistency_list}
    diff_consistency_ratio = {format_pair: (round(diff_consistency_ratio[format_pair][0], 5), diff_consistency_ratio[format_pair][1]) for format_pair in diff_consistency_ratio}
    # print("Token diff -> consistency ratio:")
    
    print("pair".ljust(15), "same".ljust(15), "diff".ljust(15))
    format_pairs = sorted(list(set(list(same_len_consistency_list.keys()) + list(diff_len_consistency_list.keys()))))
    for format_pair in format_pairs:
        print(str(format_pair).ljust(15), 
              f"{same_consistency_ratio.get(format_pair, (0, 0))}".ljust(15), 
              f"{diff_consistency_ratio.get(format_pair, (0, 0))}".ljust(15),
            )
    # B) When tokenized same, how often are answers same? analysis on minimum token length
    min_len_consistency_ratio = (sum(min_len_consistency_list)/len(min_len_consistency_list), len(min_len_consistency_list))
    min_len_consistency_ratio = (round(min_len_consistency_ratio[0], 5), min_len_consistency_ratio[1])
    print("When tokenized same")
    print(min_len_consistency_ratio)

    # C) When tokenized different, how often are answers same? analysis on random token length
    ran_consistency_ratio = (sum(ran_consistency_list)/len(ran_consistency_list), len(ran_consistency_list))
    ran_consistency_ratio = (round(ran_consistency_ratio[0], 5), ran_consistency_ratio[1])
    print("When tokenized different")
    print(ran_consistency_ratio)

    # 1) Point-biserial correlation: same_answer (binary) vs. token_diff (discrete..)
    corr, p_value = pointbiserialr(df['same_answer'], df['token_diff'])
    point_biserial = {
        "corr": corr,
        "p_value": p_value
    }
    print(f"Point-biserial correlation = {corr:.4f}, p-value = {p_value:.4e}")

    # 2) Logistic Regression
    X = df[['token_diff']].values  # predictor
    y = df['same_answer'].values   # binary outcome
    model = LogisticRegression().fit(X, y)
    print("LogisticRegression Coeff:", model.coef_[0], "Intercept:", model.intercept_)
    print("LogisticRegression Score (accuracy):", model.score(X, y))
    logistic_regression = {
        "coef": model.coef_[0][0],
        "intercept": model.intercept_[0],
        "score": model.score(X, y)
    }
    # X = df[["same_answer"]].values  # predictor
    # y = df['token_diff_binary'].values   # binary outcome
    # model = LogisticRegression().fit(X, y)
    # print("LogisticRegression Coeff:", model.coef_[0], "Intercept:", model.intercept_)
    # print("LogisticRegression Score (accuracy):", model.score(X, y))
    # logistic_regression = {
    #     "coef": model.coef_[0][0],
    #     "intercept": model.intercept_[0],
    #     "score": model.score(X, y)
    # }


    # 3) Grouped analysis: token_diff bins
    bins = [0, 1, 2, 3]  # adjust as you see fit
    df['token_diff_bin'] = pd.cut(df['token_diff'], bins=bins)
    group_stats = df.groupby('token_diff_bin',observed=False)['same_answer'].mean()
    print("Mean consistency by token_diff_bin:")
    print(group_stats)
    grouped_analysis = {
        "mean_consistency": str(group_stats)
    }

    # Visualize
    group_stats.plot(kind='bar', ylabel='Fraction same_answer', xlabel='Token difference bin',
                     title='Consistency vs. Token Difference (Micro-level)')
    import matplotlib.pyplot as plt
    plt.savefig(os.path.join(os.path.dirname(tokenlen_path), f"{prompting_strategy}_grouped_token_consistency_chart.pdf"))
    plt.show()

    # 4) Phi coefficient
    phi, p_value = pearsonr(df['token_diff_binary'], df['same_answer'])
    print(f"Phi coefficient = {phi:.4f}, p-value = {p_value:.4e}")
    phi_coefficient = {
        "phi": phi,
        "p_value": p_value
    }

    # 5) Return results
    results = {
        "point_biserial": point_biserial,
        "logistic_regression": logistic_regression,
        "grouped_analysis": grouped_analysis,
        "phi_coefficient": phi_coefficient
    }
    with open(output_path, 'w') as fout:
        json.dump(results, fout, indent=4)

def run_macro_analysis_broken(tokenlen_path, answers_path, output_dir, formats_mean_std, min_token_format):
    if formats_mean_std is None:
        with jsonlines.open(tokenlen_path) as fin:
            formats_mean_std = defaultdict(int)
            len = 0
            for record in fin:
                len += 1
                for f, token_len in record["diff_token_len"].items():
                    formats_mean_std["format"+f] += token_len
            formats_mean_std = {k: (v/len, -1) for k, v in formats_mean_std.items()}

    with open(output_dir+"_score.json", "r") as fin:
        score = json.load(fin)
    consistency = score["consistency"]
    pairwise_consistency = {
        ('format0', 'format1'): consistency['0_1'], ("format0", "format2"): consistency['0_2'], ("format0", "format3"): consistency['0_3'], ("format0", "format4"): consistency['0_4'], ("format0", "format5"): consistency['0_5'], ("format0", "format6"): consistency['0_6'], ("format0", "format7"): consistency['0_7'],
        ('format1', 'format2'): consistency['1_2'], ("format1", "format3"): consistency['1_3'], ("format1", "format4"): consistency['1_4'], ("format1", "format5"): consistency['1_5'], ("format1", "format6"): consistency['1_6'], ("format1", "format7"): consistency['1_7'],
        ('format2', 'format3'): consistency['2_3'], ("format2", "format4"): consistency['2_4'], ("format2", "format5"): consistency['2_5'], ("format2", "format6"): consistency['2_6'], ("format2", "format7"): consistency['2_7'],
        ('format3', 'format4'): consistency['3_4'], ("format3", "format5"): consistency['3_5'], ("format3", "format6"): consistency['3_6'], ("format3", "format7"): consistency['3_7'],
        ('format4', 'format5'): consistency['4_5'], ("format4", "format6"): consistency['4_6'], ("format4", "format7"): consistency['4_7'],
        ('format5', 'format6'): consistency['5_6'], ("format5", "format7"): consistency['5_7'],
        ('format6', 'format7'): consistency['6_7']
    }

    x_vals, y_vals = [], []

    for (f1, f2), val in pairwise_consistency.items():
        # 1. discrete mapping
        # If both formats share the same mapping, treat it as group 0; otherwise group 1
        group = 0 if int(formats_mean_std[f1][0]) == int(formats_mean_std[f2][0]) else 1

        # 2. continuous mapping
        # differce between mapping of f1 and f2
        group = abs(formats_mean_std[f1][0] - formats_mean_std[f2][0])

        x_vals.append(group)
        y_vals.append(val)

    # Create the plot with broken x-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [1, 1]})

    # Plot data in the first axis (smaller group differences)
    ax1.scatter(x_vals, y_vals)
    ax1.set_xlim(-0.05, 0.1)  # Adjust the range for the first axis
    ax1.set_xlabel('Similar length group')
    ax1.set_ylabel('Pairwise Consistency')
    ax1.grid(True)

    # Plot data in the second axis (larger group differences)
    ax2.scatter(x_vals, y_vals)
    ax2.set_xlim(0.9, 1.05)  # Adjust the range for the second axis
    ax2.set_xlabel('Different length group')
    ax2.grid(True)

    # Add the broken axis effect
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()
    ax2.tick_params(labelleft=False)  # Hide labels on the second y-axis

    # Add slashes for broken axis
    d = 0.02  # Size of the slashes
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    # Add slash to the right of ax1
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Top-right
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Bottom-right

    kwargs.update(transform=ax2.transAxes)  # Use ax2 transform for second axis
    # Add slash to the left of ax2
    ax2.plot((-d, +d), (-d, +d), **kwargs)  # Top-left
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Bottom-left

    plt.suptitle(f'Format Pair Group vs. Pairwise Consistency', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir+"_pairwise_tokenlen_consistency_scatter.pdf")
    plt.show()
    print(x_vals)

def run_macro_analysis(tokenlen_path, answers_path, output_dir, formats_mean_std, min_token_format):
    if formats_mean_std is None:
        with jsonlines.open(tokenlen_path) as fin:
            formats_mean_std = defaultdict(int)
            len = 0
            for record in fin:
                len += 1
                for f, token_len in record["diff_token_len"].items():
                    formats_mean_std["format"+f] += token_len
            formats_mean_std = {k: (v/len, -1) for k, v in formats_mean_std.items()}

    with open(output_dir+"_score.json", "r") as fin:
        score = json.load(fin)
    consistency = score["consistency"]
    pairwise_consistency = {
        ('format0', 'format1'): consistency['0_1'], ("format0", "format2"): consistency['0_2'], ("format0", "format3"): consistency['0_3'], ("format0", "format4"): consistency['0_4'], ("format0", "format5"): consistency['0_5'], ("format0", "format6"): consistency['0_6'], ("format0", "format7"): consistency['0_7'],
        ('format1', 'format2'): consistency['1_2'], ("format1", "format3"): consistency['1_3'], ("format1", "format4"): consistency['1_4'], ("format1", "format5"): consistency['1_5'], ("format1", "format6"): consistency['1_6'], ("format1", "format7"): consistency['1_7'],
        ('format2', 'format3'): consistency['2_3'], ("format2", "format4"): consistency['2_4'], ("format2", "format5"): consistency['2_5'], ("format2", "format6"): consistency['2_6'], ("format2", "format7"): consistency['2_7'],
        ('format3', 'format4'): consistency['3_4'], ("format3", "format5"): consistency['3_5'], ("format3", "format6"): consistency['3_6'], ("format3", "format7"): consistency['3_7'],
        ('format4', 'format5'): consistency['4_5'], ("format4", "format6"): consistency['4_6'], ("format4", "format7"): consistency['4_7'],
        ('format5', 'format6'): consistency['5_6'], ("format5", "format7"): consistency['5_7'],
        ('format6', 'format7'): consistency['6_7']
    }

    x_vals, y_vals = [], []

    for (f1, f2), val in pairwise_consistency.items():
        # 1. discrete mapping
        # If both formats share the same mapping, treat it as group 0; otherwise group 1
        group = 0 if int(formats_mean_std[f1][0]) == int(formats_mean_std[f2][0]) else 1

        # 2. continuous mapping
        # differce between mapping of f1 and f2
        group = abs(formats_mean_std[f1][0] - formats_mean_std[f2][0])

        x_vals.append(group)
        y_vals.append(val)
    plt.close('all')
    plt.scatter(x_vals, y_vals)
    plt.xlabel('Absolute Difference in Mean Token Length')
    plt.ylabel('Pairwise Consistency')
    plt.xlim(min(x_vals) - 0.1, max(x_vals) + 0.1)  # Adjust the limits to fit your data
    plt.grid(True)
    plt.title('Pairwise Consistency vs. Token Length Difference')
    plt.tight_layout()
    plt.savefig(output_dir + "_pairwise_tokenlen_consistency_scatter.pdf")
    plt.show()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Debug-DeepSeek-8B-R1")
    parser.add_argument("--dataset_name", type=str, default="QASC")
    parser.add_argument("--prompting_strategy", type=str, default="few-shot")
    args = parser.parse_args()
    model_name, dataset_name, prompting_strategy = args.model_name, args.dataset_name, args.prompting_strategy
    #TODO: results_shared_json_v1 확인요망!
    output_dir = f"results/{dataset_name}/{model_name}/{prompting_strategy}"
    formats_mean_std, min_token_format = None, None
    if not os.path.exists(f"results/{dataset_name}/{model_name}/{prompting_strategy}_tokenlen.jsonl"):
        print("Token length analysis not found. Running token length analysis...")
        formats_mean_std, min_token_format =  analyze_token_length(model_name=model_name, task_name=dataset_name, prompting_strategy=prompting_strategy)
    tokenlen_path=f"results/{dataset_name}/{model_name}/{prompting_strategy}_tokenlen.jsonl"
    answers_path=f"results/{dataset_name}/{model_name}/{prompting_strategy}_predictions.jsonl"
    output_path=f"results/{dataset_name}/{model_name}/{prompting_strategy}_micro_analysis.json"
    run_micro_analysis(tokenlen_path, answers_path, output_path)
    run_macro_analysis(tokenlen_path, answers_path, output_dir, formats_mean_std, min_token_format)