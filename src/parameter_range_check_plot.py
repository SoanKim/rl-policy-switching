#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim)
# Title: parameter_range_check_plot.py
# Explanation: Diagnostic tool that imports the exact search space from run_fitting.py

##### DATA FILES ARE UNAVAILABLE UNTIL THE PAPER PUBLISHED #####

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys


try:
    sys.path.append(os.getcwd())
    from run_fitting import SPACE as CHAPTER_3_SPACE

    print(f"Successfully imported parameter space from run_fitting.py ({len(CHAPTER_3_SPACE)} parameters)")
except ImportError as e:
    print(f"Error: Could not import 'SPACE' from run_fitting.py.\nDetails: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error during import: {e}")
    sys.exit(1)


def plot_parameter_histograms(df, space, output_filename):
    available_params = [s for s in space if s.name in df.columns]

    if not available_params:
        print(f"Warning: None of the expected parameters were found in the CSV.")
        return

    param_names = [s.name for s in available_params]
    n_params = len(param_names)

    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i, param_name in enumerate(param_names):
        ax = axes[i]
        param_space = next((s for s in available_params if s.name == param_name), None)
        low, high = param_space.low, param_space.high

        data = df[param_name]

        if data.nunique() <= 1:
            sns.histplot(data, ax=ax, kde=False, bins=1, color='skyblue', edgecolor='black')
        else:
            sns.histplot(data, ax=ax, kde=True, bins=15, color='skyblue', edgecolor='black')

        ax.set_title(f'{param_name}', fontsize=12, fontweight='bold', color='black')
        ax.set_xlabel('Fitted Value')

        ax.axvline(low, color='r', linestyle='--', linewidth=2, label=f'LB: {low}')
        ax.axvline(high, color='r', linestyle='--', linewidth=2, label=f'UB: {high}')

        ax.legend()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"Parameter range plot saved to: {output_filename}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot fitted parameter distributions.")
    parser.add_argument("--fitted-file", required=True, help="Path to the fitted CSV.")
    parser.add_argument("--output-file", required=True, help="Output path for the PNG.")

    args = parser.parse_args()

    if not os.path.exists(args.fitted_file):
        sys.exit(f"Error: File not found at '{args.fitted_file}'")

    try:
        results_df = pd.read_csv(args.fitted_file)
    except Exception as e:
        sys.exit(f"Error reading CSV: {e}")

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plot_parameter_histograms(results_df, CHAPTER_3_SPACE, args.output_file)