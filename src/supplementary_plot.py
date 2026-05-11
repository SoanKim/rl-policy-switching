#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim)
# Title: supplementary_plot.py

##### DATA FILES ARE UNAVAILABLE UNTIL THE PAPER PUBLISHED #####

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

os.makedirs('Chapter_3_Fig', exist_ok=True)

try:
    df_param = pd.read_csv('results/fitting/parameter_recovery_results.csv')

    params = [
        ('learning_rate', r'Learning rate ($\alpha$)'),
        ('decay_rate', r'Decay rate ($\phi$)'),
        ('beta_softmax', r'Inverse temp. ($\beta$)'),
        ('k_gain', r'Gating sensitivity ($k$)'),
        ('theta_threshold', r'Gating threshold ($\theta$)'),
        ('epsilon_leak', r'Baseline att. ($\epsilon$)')
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (col, name) in enumerate(params):
        ax = axes[i]
        true_col = f'true_{col}'
        rec_col = f'recovered_{col}'

        if true_col in df_param.columns and rec_col in df_param.columns:
            sns.regplot(x=df_param[true_col], y=df_param[rec_col], ax=ax,
                        scatter_kws={'alpha': 0.5, 'color': 'skyblue'}, line_kws={'color': 'red'})

            r_val, p_val = pearsonr(df_param[true_col], df_param[rec_col])
            ax.text(0.05, 0.95, f'$r$ = {r_val:.2f}', transform=ax.transAxes,
                    fontsize=12, verticalalignment='top')

            ax.set_title(name, fontsize=14)
            ax.set_xlabel('True Parameter')
            ax.set_ylabel('Recovered Parameter')

    plt.suptitle('Parameter Recovery: Policy Switching', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('Chapter_3_Fig/parameter_recovery_policy_switching.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Successfully generated parameter recovery plot.")

except FileNotFoundError:
    print("Skipping parameter recovery plot: parameter_recovery_results.csv not found.")

try:
    df_model = pd.read_csv('results/fitting/model_recovery_results.csv')
    confusion_matrix = df_model.pivot(index='true_model', columns='fitted_model', values='proportion')

    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt=".2f", vmin=0, vmax=1)

    plt.title('Model Recovery Confusion Matrix', fontsize=14, pad=15)
    plt.ylabel('True Generative Model', fontsize=12)
    plt.xlabel('Best Fitting Model (by BIC)', fontsize=12)

    plt.tight_layout()
    plt.savefig('Chapter_3_Fig/model_recovery.png', dpi=300)
    plt.close()
    print("Successfully generated model recovery plot.")

except FileNotFoundError:
    print("Skipping model recovery plot: model_recovery_results.csv not found.")