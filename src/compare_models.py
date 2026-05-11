#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Soan Kim (https://github.com/SoanKim)
# Title: compare_models.py
# Explanation: Comparison & Diagnostic Suite.

##### DATA FILES ARE UNAVAILABLE UNTIL THE PAPER PUBLISHED #####

import pandas as pd
import numpy as np
import os
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import entropy
from tqdm import tqdm


try:
    from agent import DualProcessAgent
    from metacontroller import ThalamicMetacontroller
    from env import OddOneOutEnv
except ImportError:
    print("Critical Error: Could not import 'agent', 'metacontroller', or 'env'.")
    exit(1)

sns.set_style("whitegrid")
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.family'] = 'sans-serif'


def format_mean_sd(x):
    return f"{x.mean():.2f} ({x.std():.2f})"


def get_equations():
    return r"""
\section*{Computational Model Definitions}
\subsection*{1. Gating Function}
\begin{equation} \lambda(PE) = \epsilon + \frac{1 - \epsilon}{1 + e^{-k(PE - \theta)}} \end{equation}

\subsection*{2. Model A: Signal Enhancement}
\begin{align} Q(a) &= w_{id} \cdot \phi_{id}(a) + w_{div} \cdot (\phi_{div}(a) \cdot \lambda(PE)) \\ \pi(a|s) &= \text{Softmax}(\beta \cdot Q(a)) \end{align}

\subsection*{3. Model B: Policy Switching}
\begin{align} \pi_{final}(a|s) &= (1 - \lambda(PE)) \cdot \pi_{id}(a|s) + \lambda(PE) \cdot \pi_{div}(a|s) \end{align}
"""


def save_full_report(output_dir, df_res, full_df, df_diag):
    t1 = df_res[['Model', 'Total AIC', 'Total BIC', 'Mean NLL', 'Delta AIC', 'Delta BIC']].copy()
    t1.columns = ['Model', 'Total AIC', 'Total BIC', 'Mean NLL', '$\Delta$ AIC', '$\Delta$ BIC']
    tex_t1 = t1.to_latex(index=False, formatters={'Total AIC': '{:,.2f}'.format, 'Total BIC': '{:,.2f}'.format,
                                                  'Mean NLL': '{:,.2f}'.format, '$\Delta$ AIC': '{:,.2f}'.format,
                                                  '$\Delta$ BIC': '{:,.2f}'.format},
                         caption='Model Comparison (AIC & BIC)', label='tab:model_comparison', escape=False)

    t2 = full_df.groupby('Model')[
        ['learning_rate', 'decay_rate', 'beta_softmax', 'k_gain', 'theta_threshold', 'epsilon_leak']].apply(
        lambda x: x.apply(format_mean_sd)).reset_index()
    t2.columns = ['Model', r'$\alpha$', r'$\phi$', r'$\beta$', r'$k$', r'$\theta$', r'$\epsilon$']
    tex_t2 = t2.to_latex(index=False, caption='Estimated Parameters (Mean $\pm$ SD)', label='tab:parameters',
                         escape=False)

    t3 = df_diag.groupby('Model')[['Entropy', 'P_Target', 'P_Distractor']].apply(
        lambda x: x.apply(format_mean_sd)).reset_index()
    t3.columns = ['Model', 'Entropy', 'P(Target)', 'P(Blocker)']
    tex_t3 = t3.to_latex(index=False, caption='Internal Dynamics (Mean $\pm$ SD)', label='tab:diagnostics',
                         escape=False)

    with open(os.path.join(output_dir, "model_comparison.tex"), 'w') as f:
        f.write(get_equations() + "\n\n" + tex_t1 + "\n\n" + tex_t2 + "\n\n" + tex_t3)
    print(f"LaTeX Report saved to {output_dir}/model_comparison.tex")


def run_probe_simulation(row):
    """Simulates agent on Practice -> Critical Probe."""
    model_key = 'policy_switching' if 'Policy' in row['Model'] else 'signal_enhancement'
    agent = DualProcessAgent(row['learning_rate'], row['decay_rate'], row['beta_softmax'], model_key)
    meta = ThalamicMetacontroller(row['k_gain'], row['theta_threshold'], row['epsilon_leak'])
    env = OddOneOutEnv()

    np.random.seed(42)
    for _ in range(36):
        obs, pe, target, _ = env.reset(phase='practice', condition='standard')
        gate = meta.get_gating_value(0.0)
        probs = agent.get_action_probs(obs, gate)
        action = np.random.choice(4, p=probs)
        agent.update(action, 1.0 if action == target else 0.0)

    f_id = np.array([0.0, 1.0, 0.0, 0.0])
    f_div = np.array([1.0, 0.0, 0.0, 0.0])
    gate = meta.get_gating_value(1.0)
    probs = agent.get_action_probs(gate_value=gate, external_features=(f_id, f_div))

    return {'Model': row['Model'], 'Entropy': entropy(probs, base=2), 'P_Target': probs[0], 'P_Distractor': probs[1]}


def get_sensitivity_curve(row):
    """Simulates response to PE varying 0->1."""
    model_key = 'policy_switching' if 'Policy' in row['Model'] else 'signal_enhancement'
    agent = DualProcessAgent(row['learning_rate'], row['decay_rate'], row['beta_softmax'], model_key)
    meta = ThalamicMetacontroller(row['k_gain'], row['theta_threshold'], row['epsilon_leak'])
    agent.w_id, agent.w_div = 0.8, 0.2

    curve = []
    for pe in np.linspace(0, 1, 20):
        gate = meta.get_gating_value(pe)
        probs = agent.get_action_probs(gate_value=gate,
                                       external_features=(np.array([0, 1, 0, 0]), np.array([1, 0, 0, 0])))
        curve.append({'Model': row['Model'], 'PE': pe, 'P_Target': probs[0]})
    return curve


def get_palette(full_df, winner_name):
    return {m: '#FFFFFF' if m == winner_name else '#808080' for m in full_df['Model'].unique()}


def apply_hatch(ax):
    for patch in ax.patches:
        if patch.get_facecolor()[:3] == (1.0, 1.0, 1.0): patch.set_hatch('//'); patch.set_edgecolor('black')
    for collection in ax.collections:
        if collection.get_facecolor().mean() > 0.9: collection.set_hatch('//'); collection.set_edgecolor('black')
        collection.set_edgecolor('black')


def plot_aic_bic(df_res, output_dir, winner_name):
    plt.figure(figsize=(8, 6))
    df_melt = df_res.melt(id_vars='Model', value_vars=['Total AIC', 'Total BIC'], var_name='Metric', value_name='Score')
    sns.barplot(data=df_melt, x='Metric', y='Score', hue='Model', palette=get_palette(df_res, winner_name),
                edgecolor='black')
    apply_hatch(plt.gca())
    plt.ylim(df_melt['Score'].min() * 0.95, df_melt['Score'].max() * 1.02)
    plt.title("Model Selection (AIC/BIC)", fontsize=16, fontweight='bold', pad=20)
    plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_selection.png"), dpi=300)
    plt.close()


def plot_parameters(full_df, output_dir, winner_name):
    print("   > Plotting Parameters...")
    params = [('epsilon_leak', r'$\epsilon$ (Leak)'), ('k_gain', r'$k$ (Gain)'),
              ('theta_threshold', r'$\theta$ (Thresh)'),
              ('beta_softmax', r'$\beta$ (Temp)'), ('learning_rate', r'$\alpha$ (LR)'),
              ('decay_rate', r'$\phi$ (Decay)')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, (col, lbl) in enumerate(params):
        sns.violinplot(data=full_df, x='Model', y=col, ax=axes[i], palette=get_palette(full_df, winner_name),
                       inner='box', hue='Model', legend=False)
        apply_hatch(axes[i])
        axes[i].set_title(lbl, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
    plt.suptitle("Hyperparameter Distributions", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parameter_violins.png"), dpi=300)
    plt.close()


def plot_diagnostics(full_df, df_diag, output_dir, winner_name):
    print("   > Plotting Mechanisms...")
    palette = get_palette(full_df, winner_name)

    plt.figure(figsize=(6, 5))
    sns.violinplot(data=df_diag, x='Model', y='Entropy', palette=palette, hue='Model', legend=False)
    apply_hatch(plt.gca())
    plt.title("Decision Uncertainty", fontweight='bold');
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mechanism_entropy.png"), dpi=300);
    plt.close()

    median_params = full_df.groupby('Model').median(numeric_only=True).reset_index()
    curves = []
    for _, row in median_params.iterrows(): curves.extend(get_sensitivity_curve(row))
    df_curve = pd.DataFrame(curves)

    plt.figure(figsize=(6, 5))
    for m in df_curve['Model'].unique():
        sub = df_curve[df_curve['Model'] == m]
        style = '-' if m == winner_name else '--'
        color = 'black' if m == winner_name else '#666666'
        plt.plot(sub['PE'], sub['P_Target'], linestyle=style, color=color, linewidth=3, label=m)
    plt.title("PE Sensitivity", fontweight='bold')
    plt.xlabel("PE Signal")
    plt.ylabel("P(Diversity)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mechanism_sensitivity.png"), dpi=300)
    plt.close()

    df_melt = df_diag.melt(id_vars='Model', value_vars=['P_Target', 'P_Distractor'], var_name='Option',
                           value_name='Prob')
    plt.figure(figsize=(8, 6))

    chart = sns.barplot(data=df_melt, x='Option', y='Prob', hue='Model', palette=palette, edgecolor='black',
                        errorbar='se')
    apply_hatch(plt.gca())

    plt.title("Conflict Resolution (Target vs Blocker)", fontweight='bold', fontsize=16, pad=20)
    plt.ylabel("Selection Probability", fontsize=14, fontweight='bold')
    plt.xlabel("Choice Option", fontsize=14, fontweight='bold')

    chart.set_xticklabels(['Target (Diversity)', 'Distractor (Identity)'], fontsize=12)

    plt.legend(title=None, frameon=False, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mechanism_conflict.png"), dpi=300);
    plt.close()


def compare(fitting_dir, output_dir):
    print("=== FULL THESIS REPORT GENERATION ===")

    models = ['policy_switching', 'signal_enhancement']
    dfs = [pd.read_csv(os.path.join(fitting_dir, f"fitted_surprise_exp2_{m}.csv")).assign(
        Model=m.replace('_', ' ').title())
           for m in models if os.path.exists(os.path.join(fitting_dir, f"fitted_surprise_exp2_{m}.csv"))]
    if not dfs: return
    full_df = pd.concat(dfs)

    k, n = 6, 108 * 50
    df_res = full_df.groupby('Model')[['nll']].sum().reset_index()
    df_res['Total AIC'] = 2 * k + 2 * df_res['nll']
    df_res['Total BIC'] = k * np.log(n) + 2 * df_res['nll']
    df_res['Mean NLL'] = full_df.groupby('Model')['nll'].mean().values
    df_res['Delta AIC'] = df_res['Total AIC'] - df_res['Total AIC'].min()
    df_res['Delta BIC'] = df_res['Total BIC'] - df_res['Total BIC'].min()
    winner_name = df_res.loc[df_res['Total BIC'].idxmin()]['Model']
    print(f"🏆 Winner: {winner_name}")

    print("   > Running Simulations...")
    df_diag = pd.DataFrame([run_probe_simulation(row) for _, row in tqdm(full_df.iterrows(), total=len(full_df))])

    os.makedirs(output_dir, exist_ok=True)
    save_full_report(output_dir, df_res, full_df, df_diag)
    plot_aic_bic(df_res, output_dir, winner_name)
    plot_parameters(full_df, output_dir, winner_name)
    plot_diagnostics(full_df, df_diag, output_dir, winner_name)
    print(f"All files saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit_dir', default='results/fitting')
    parser.add_argument('--out_dir', default='results/plots')
    args = parser.parse_args()
    compare(args.fit_dir, args.out_dir)