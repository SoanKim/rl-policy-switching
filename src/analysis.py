#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim)
# Title: analysis.py
# Explanation: Generates plots for the 3-phase structure (Practice -> Test Block 1 -> Test Block 2).
##### DATA FILES ARE UNAVAILABLE UNTIL THE PAPER PUBLISHED #####

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import os
import numpy as np
from scipy.stats import ttest_ind

sns.set_style("whitegrid")
sns.despine()

COLOR_PALETTE = {
    'Identity': 'grey',
    'Diversity (Practice)': 'purple',
    'Diversity (Standard)': 'blue',
    'Diversity (Oddball)': 'red'
}

PHASE_MAP = {
    'Block1': 'Test Block 1',
    'Block2': 'Test Block 2',
    'Practice': 'Practice'
}


def get_star_string(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'n.s.'


def add_stat_bracket(ax, x1, x2, y, text):
    h = 0.02
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='k')
    ax.text((x1 + x2) * 0.5, y + h, text, ha='center', va='bottom', color='k', fontsize=12)


def save_plot(fig, output_dir, model_name, filename):
    name = f"{model_name}_{filename}"
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_stats_table(df_agg, output_dir, model_name):
    print(f"Generating Stats Table for {model_name}...")
    stats_df = df_agg[
        (df_agg['Phase'].isin(['Test Block 1', 'Test Block 2'])) &
        (df_agg['TrialType'] == 'Diversity')
        ].copy()

    summary = stats_df.groupby(['Phase', 'Condition'])['Accuracy'].agg(['mean', 'sem']).reset_index()
    summary['Result'] = summary.apply(lambda x: f"{x['mean']:.2f} ($\pm${x['sem']:.2f})", axis=1)

    pivot_table = summary.pivot(index='Condition', columns='Phase', values='Result')
    cols = ['Test Block 1', 'Test Block 2']
    pivot_table = pivot_table[cols]

    latex_code = pivot_table.to_latex(
        caption=f"Simulation Results ({model_name.replace('_', ' ').title()}): Mean Accuracy ($\pm$ SEM).",
        label=f"tab:sim_stats_{model_name}",
        escape=False
    )

    with open(os.path.join(output_dir, f"{model_name}_simulation_stats.tex"), 'w') as f:
        f.write(latex_code)
    print("✅ Stats table saved.")


def plot_test_results(df_agg, output_dir, model_name):
    """
    Fig 1: Bar Plot comparing Block 1 vs Block 2 for Diversity Trials.
    Updated: Legend now says "Diversity (Standard)" and "Diversity (Oddball)".
    """
    print(f"Generating Fig 1 for {model_name}...")

    df_viz = df_agg[
        (df_agg['Phase'].isin(['Test Block 1', 'Test Block 2'])) &
        (df_agg['TrialType'] == 'Diversity')
        ].copy()

    plt.figure(figsize=(8, 6))

    bar_palette = {'Standard': '#FFFFFF', 'Oddball': '#808080'}

    ax = sns.barplot(
        data=df_viz, x='Phase', y='Accuracy', hue='Condition',
        palette=bar_palette, capsize=.1, errorbar='se',
        order=['Test Block 1', 'Test Block 2'],
        edgecolor='black', linewidth=1.2,
        err_kws={'linewidth': 1.5, 'color': 'black'}
    )

    for i, phase in enumerate(['Test Block 1', 'Test Block 2']):
        std = df_viz[(df_viz['Phase'] == phase) & (df_viz['Condition'] == 'Standard')]['Accuracy']
        odd = df_viz[(df_viz['Phase'] == phase) & (df_viz['Condition'] == 'Oddball')]['Accuracy']

        if len(std) > 1 and len(odd) > 1:
            _, p_val = ttest_ind(std, odd, equal_var=False)
            star = get_star_string(p_val)

            m_std, s_std = std.mean(), std.sem()
            m_odd, s_odd = odd.mean(), odd.sem()
            bar_top = max(m_std + s_std, m_odd + s_odd)

            add_stat_bracket(ax, i - 0.2, i + 0.2, bar_top + 0.05, star)

    plt.title(f"{model_name.replace('_', ' ').title()}: Unblocking Effect", fontsize=16, fontweight='bold', pad=20)
    plt.ylabel("Mean Accuracy", fontsize=14)
    plt.ylim(0, 1.15)
    plt.axhline(0.25, linestyle=':', color='black', alpha=0.7, label='Chance')

    legend_handles = [
        mpatches.Patch(facecolor='#FFFFFF', edgecolor='black', label='Diversity (Standard)'),
        mpatches.Patch(facecolor='#808080', edgecolor='black', label='Diversity (Oddball)')
    ]
    plt.legend(handles=legend_handles, loc='upper right', frameon=True)

    sns.despine()
    plt.tight_layout()
    save_plot(plt.gcf(), output_dir, model_name, "block_comparison.png")


def plot_learning_curves(df, output_dir, model_name):
    """
    Fig 2: Continuous Timeline (Colored Lines).
    Legend now displays: Identity, Diversity (Practice), Diversity (Standard), Diversity (Oddball).
    """
    print(f"Generating Fig 2: Timeline ({model_name})...")
    df_curve = df.copy()

    df_curve['Phase'] = df_curve['Phase'].map(PHASE_MAP).fillna(df_curve['Phase'])

    df_curve['Condition'] = df_curve['Condition'].replace({
        'Standard': 'Diversity (Standard)',
        'Oddball': 'Diversity (Oddball)'
    })

    mask_prac = df_curve['Phase'] == 'Practice'
    df_curve.loc[mask_prac, 'Condition'] = 'Diversity (Practice)'

    plt.figure(figsize=(12, 7))

    sns.lineplot(data=df_curve[df_curve['TrialType'] == 'Identity'],
                 x='Trial', y='Accuracy', color='grey',
                 linestyle='--', linewidth=2, errorbar=None, label='Identity')

    sns.lineplot(data=df_curve[df_curve['TrialType'] == 'Diversity'],
                 x='Trial', y='Accuracy', hue='Condition',
                 palette=COLOR_PALETTE, linewidth=2.5, errorbar='se')

    max_prac = 36
    max_b1 = 72

    plt.axvline(max_prac, color='k', linestyle=':', alpha=0.5)
    plt.axvline(max_b1, color='k', linestyle=':', alpha=0.5)

    plt.text(max_prac / 2, 1.05, 'Practice', ha='center', fontsize=12)
    plt.text(max_prac + (max_b1 - max_prac) / 2, 1.05, 'Test Block 1', ha='center', fontsize=12)
    plt.text(max_b1 + 10, 1.05, 'Test Block 2', ha='center', fontsize=12)

    plt.title(f"{model_name.replace('_', ' ').title()}: Timeline", fontsize=16, pad=45)
    plt.ylim(0, 1.15)
    plt.axhline(0.25, linestyle=':', color='black')

    plt.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, 1.08),
               ncol=4, frameon=False, fontsize=11)

    save_plot(plt.gcf(), output_dir, model_name, "timeline.png")


def plot_gate_mechanism(df, output_dir, model_name):
    print(f"Generating Fig 3: Gate ({model_name})...")

    df_gate = df[df['Phase'].isin(['Test Block 1', 'Test Block 2'])].copy()

    if df_gate.empty:
        print("⚠️ Warning: Gate DataFrame is empty! Check phase names.")
        return

    plt.figure(figsize=(10, 5))

    sns.lineplot(
        data=df_gate,
        x='Trial',
        y='Gate',
        hue='Condition',
        style='Condition',
        markers=False,
        dashes={'Standard': (2, 2), 'Oddball': (1, 0)},
        palette={'Standard': 'black', 'Oddball': 'black'},
        linewidth=2.5,
        errorbar=None
    )

    plt.title(f"{model_name.replace('_', ' ').title()}: Gate Activation", fontsize=16, fontweight='bold', pad=40)
    plt.ylabel("Gate Value (Attention)", fontsize=14, fontweight='bold')
    plt.xlabel("Trial", fontsize=14, fontweight='bold')

    plt.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, 1.12),
               fontsize=11, ncol=2, frameon=False)

    sns.despine()
    plt.tight_layout()
    save_plot(plt.gcf(), output_dir, model_name, "gate_activation.png")


def plot_bar_comparison(df_agg, output_dir, model_name):
    print(f"Generating Fig 2: Bar Comparison ({model_name})...")
    df_viz = df_agg.copy()

    def get_bar_group(row):
        if row['TrialType'] == 'Identity': return 'Identity'
        if row['Phase'] == 'Practice': return 'Diversity (Standard)'
        if row['Condition'] == 'Standard': return 'Diversity (Standard)'
        return 'Diversity (Oddball)'

    df_viz['BarGroup'] = df_viz.apply(get_bar_group, axis=1)

    bar_palette = {
        'Identity': '#FFFFFF',
        'Diversity (Standard)': '#FFFFFF',
        'Diversity (Oddball)': '#808080'
    }

    plt.figure(figsize=(10, 8))
    order = ['Practice', 'Test Block 1', 'Test Block 2']
    hue_order = ['Identity', 'Diversity (Standard)', 'Diversity (Oddball)']

    ax = sns.barplot(
        data=df_viz, x='Phase', y='Accuracy', hue='BarGroup',
        palette=bar_palette, order=order, hue_order=hue_order,
        capsize=0.1, edgecolor='black', linewidth=1.2,
        errorbar='se', err_kws={'linewidth': 1.5, 'color': 'black'}
    )

    for i, patch in enumerate(ax.patches):
        if i < 3:
            patch.set_hatch('//')
            patch.set_edgecolor('black')
            patch.set_facecolor('white')

    phases_to_test = ['Test Block 1', 'Test Block 2']
    for phase in phases_to_test:
        std_data = \
        df_viz[(df_viz['Phase'] == phase) & (df_viz['Condition'] == 'Standard') & (df_viz['TrialType'] == 'Diversity')][
            'Accuracy']
        odd_data = \
        df_viz[(df_viz['Phase'] == phase) & (df_viz['Condition'] == 'Oddball') & (df_viz['TrialType'] == 'Diversity')][
            'Accuracy']

        if len(std_data) > 1 and len(odd_data) > 1:
            _, p_val = ttest_ind(std_data, odd_data, equal_var=False)
            star = get_star_string(p_val)
            m_std, s_std = std_data.mean(), std_data.sem()
            m_odd, s_odd = odd_data.mean(), odd_data.sem()
            bar_top = max(m_std + s_std, m_odd + s_odd)
            phase_idx = order.index(phase)
            add_stat_bracket(ax, phase_idx, phase_idx + 0.27, bar_top + 0.05, star)

    plt.title(f"{model_name.replace('_', ' ').title()}: Performance by Phase", fontsize=16, fontweight='bold', pad=45)
    plt.ylabel("Mean Accuracy", fontsize=14, fontweight='bold')
    plt.ylim(0, 1.15)
    plt.axhline(0.25, linestyle='--', color='gray', alpha=0.8, linewidth=1.5)

    legend_handles = [
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', label='Identity'),
        mpatches.Patch(facecolor='white', edgecolor='black', label='Diversity (Standard)'),
        mpatches.Patch(facecolor='#808080', edgecolor='black', label='Diversity (Oddball)')
    ]
    plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.05), fontsize=11, ncol=3,
               frameon=False)

    sns.despine()
    plt.tight_layout()
    save_plot(plt.gcf(), output_dir, model_name, "bar_timeline.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', default='results/plots')
    parser.add_argument('--model_name', required=True, help="Name of the model (e.g. 'policy_switching')")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found.")
        exit(1)

    df = pd.read_csv(args.input)

    if 'Condition' in df.columns:
        df['Condition'] = df['Condition'].str.capitalize()
    if 'TrialType' in df.columns:
        df['TrialType'] = df['TrialType'].str.capitalize()

    model_output_dir = os.path.join(args.out, args.model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    df['Phase'] = df['Phase'].map(PHASE_MAP).fillna(df['Phase'])

    df_agg = df.groupby(['Run', 'Condition', 'Phase', 'TrialType'])['Accuracy'].mean().reset_index()

    plot_test_results(df_agg, model_output_dir, args.model_name)
    plot_learning_curves(df, model_output_dir, args.model_name)
    plot_gate_mechanism(df, model_output_dir, args.model_name)
    plot_bar_comparison(df_agg, model_output_dir, args.model_name)
    save_stats_table(df_agg, model_output_dir, args.model_name)

    print(f"Analysis Complete. Plots and Tables saved to: {model_output_dir}")