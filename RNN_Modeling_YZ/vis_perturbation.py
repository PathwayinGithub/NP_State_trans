import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from analysis_core import TRIAL_TYPES, save_figure
from analysis_perturbation import (
    PerturbationResult,
    NOISE_WINDOW_DEFS,
    ABLATION_WINDOW_DEFS,
)


# ======================================================================
#  Style
# ======================================================================

def _set_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


FONT = dict(suptitle=13, title=11, label=10, tick=9, legend=8)

TT_LABELS = {
    'nrem_to_wake': 'N→W Transition',
    'wake_to_nrem': 'W→N Transition',
    'nrem_only':    'NREM Maintenance',
    'wake_only':    'Wake Maintenance',
}

TT_SHORT = {
    'nrem_to_wake': 'N→W',
    'wake_to_nrem': 'W→N',
    'nrem_only':    'N only',
    'wake_only':    'W only',
}


# ── 4 pathway-ablation conditions: visual encoding ─────────────────
COND_ORDER = [
    'intact',
    'ablate_n2w',
    'ablate_w2n',
]

COND_STYLES = {
    'intact': dict(
        color='#2C3E50', ls='-', lw=2.4,
        marker='o', ms=4.5, zorder=10,
        label='Intact'),
    'ablate_n2w': dict(
        color='#8E44AD', ls='-', lw=1.8,
        marker='s', ms=4.0, zorder=8,
        label='Cut N→W'),
    'ablate_w2n': dict(
        color='#2980B9', ls='-', lw=1.8,
        marker='D', ms=4.0, zorder=8,
        label='Cut W→N'),
}


# ======================================================================
#  Helpers
# ======================================================================

def _curve(summary, tt, cond, strengths, metric, n_models):
    """Extract mean ± SEM for a given metric.
    """
    vals, sems = [], []
    for s in strengths:
        d = summary.get(tt, {}).get(cond, {}).get(s, {})
        v = d.get(metric, np.nan)
        std = d.get(f'{metric}_std', 0.0)
        sem = std / np.sqrt(n_models) if n_models > 1 and std > 0 else 0.0
        vals.append(v)
        sems.append(sem)
    return np.asarray(vals, dtype=float), np.asarray(sems, dtype=float)


def _plot_all_conditions(ax, result, tt, metric, ylabel, ylim):
    """Draw all condition lines on one Axes."""
    summary   = result.summary
    strengths = result.perturbation_strengths
    n_models  = max(result.n_models, 1)

    for cond in COND_ORDER:
        if cond not in summary.get(tt, {}):
            continue
        sty = COND_STYLES[cond]
        v, se = _curve(summary, tt, cond, strengths, metric, n_models)

        ax.plot(strengths, v,
                color=sty['color'], ls=sty['ls'], lw=sty['lw'],
                marker=sty['marker'], ms=sty['ms'],
                label=sty['label'], zorder=sty['zorder'])

        if np.any(se > 0):
            ax.fill_between(strengths,
                            np.clip(v - se, 0, 1),
                            np.clip(v + se, 0, 1),
                            color=sty['color'], alpha=0.08, lw=0)

    if metric == 'accuracy':
        ax.axhline(1 / 2, color='#BDC3C7', ls=':', lw=0.8, zorder=1)

    ax.set_ylabel(ylabel, fontsize=FONT['label'])
    ax.set_ylim(ylim)
    ax.yaxis.set_major_formatter(PercentFormatter(1, 0))
    ax.tick_params(labelsize=FONT['tick'])
    ax.grid(True, alpha=0.10, lw=0.4)


# ======================================================================
#  1. Summary grid  (2 × 4)
# ======================================================================

def plot_perturbation_summary_grid(result, subtitle='', save_path=None):
    """1 row (accuracy) × 4 columns (trial types)."""
    _set_style()
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)

    for col, tt in enumerate(TRIAL_TYPES):
        ax = axes[col]
        _plot_all_conditions(
            ax, result, tt, 'accuracy',
            'Accuracy' if col == 0 else '',
            (-0.05, 1.05))

        ax.set_title(TT_SHORT[tt], fontsize=FONT['title'],
                     fontweight='bold', pad=6)
        ax.set_xlabel('Noise σ', fontsize=FONT['label'])

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles, labels,
                loc='upper right', fontsize=FONT['legend'],
                framealpha=0.92, edgecolor='none',
                handlelength=1.8, handletextpad=0.35,
                labelspacing=0.3, borderpad=0.4,
            )

    suptitle = 'Perturbation Robustness'
    if subtitle:
        suptitle += f'  —  {subtitle}'
    suptitle += f'  (n={result.n_models})'
    fig.suptitle(suptitle, fontsize=FONT['suptitle'] + 1,
                 fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_figure(fig, save_path)
    plt.close(fig)
    return fig

# ======================================================================
#  2. Entry point
# ======================================================================

def generate_all_perturbation_figures(config, all_results):
    """Generate all perturbation figures.

    Parameters
    ----------
    all_results : {abl_window: {noise_window: PerturbationResult}}

    Output directory
        {figure_dir}/perturbation/{abl_window}/{noise_window}/
    """
    figure_dir = getattr(
        config.paths, 'figure_dir',
        os.path.join(config.paths.data_dir, 'figures'))

    for aw, nw_results in all_results.items():
        abl_label = ABLATION_WINDOW_DEFS.get(aw, aw)

        for nw, result in nw_results.items():
            noise_label = NOISE_WINDOW_DEFS.get(nw, nw)
            out_dir = os.path.join(figure_dir, 'perturbation', aw, nw)

            subtitle = f'Ablation: {abl_label}  |  Noise: {noise_label}'
            print(f"\n  Figures — {subtitle}")

            # Summary grid
            plot_perturbation_summary_grid(
                result, subtitle=subtitle,
                save_path=os.path.join(out_dir, 'noise_low_to_high.png'))