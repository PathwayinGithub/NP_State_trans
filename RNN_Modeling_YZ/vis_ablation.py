import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import PercentFormatter, FormatStrFormatter, MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
from analysis_neurons import (
        _peak_entropy, _temporal_sparsity, _sequentiality_index,
    )
from analysis_core import TRIAL_TYPES, TransitionData, AblationResult, save_figure


# =====================================================================
#  Style & Colour Configuration
# =====================================================================

FONT = {
    'suptitle': 16, 'title': 14, 'label': 12,
    'tick': 11, 'legend': 10, 'annot': 10, 'bar_text': 10,
}
SUBPLOT_SIZE = 4.2

COLORS = {
    'nrem': '#E67E22', 'wake': '#3498DB',
    'n2w': '#CB4335', 'w2n': '#2874A6',
    'intact': '#566573', 'ablated': '#922B21',
}
TRIAL_COLORS = {
    'nrem_to_wake': '#CB4335', 'wake_to_nrem': '#2874A6',
    'nrem_only': '#CA6F1E', 'wake_only': '#1A5276',
}
TRIAL_MARKERS = {
    'nrem_to_wake': 'o', 'wake_to_nrem': 's',
    'nrem_only': '^', 'wake_only': 'D',
}
TRIAL_LABELS = {
    'nrem_to_wake': 'N→W', 'wake_to_nrem': 'W→N',
    'nrem_only': 'N Only', 'wake_only': 'W Only',
}
TRIAL_LABELS_SHORT = ['N→W', 'W→N', 'N Only', 'W Only']

STATE_COLORS = {'wake': '#3498DB', 'nrem': '#E67E22'}

PERF_COLORS = {'good': '#5DADE2', 'moderate': '#F5B041', 'poor': '#EC7063'}

ZONE = {
    'init_bg': '#F4F6F7',
    'delay_bg': '#FDEBD0',
    'nrem_bg': '#FDF2E9',
    'wake_bg': '#EBF5FB',
    'chance': '#ABB2B9',
    'edge': '#2C3E50',
}
CMAP_HEATMAP = 'YlOrRd'

CONDITIONS = [
    'intact', 'ablate_n2w', 'ablate_w2n', 'ablate_both',
    #'ablate_nn', 'ablate_ww',
]
COND_TITLES = [
    'Intact', 'Cut N→W Path', 'Cut W→N Path', 'Cut Both Trans.',
    #'Cut N→N Path', 'Cut W→W Path',
]
COND_LABELS_MAP = {
    'intact': 'Intact',
    'ablate_n2w': 'Cut N→W',
    'ablate_w2n': 'Cut W→N',
    'ablate_both': 'Cut Both Trans.',
    # 'ablate_all': 'Cut All',
    # 'ablate_nn': 'Cut N→N',
    # 'ablate_ww': 'Cut W→W',
}

# Temporal-window constants
WINDOW_KEYS = ['init_delay',
    #'full_trial', 'init', 'delay', 'init_delay', 'response', 'delay_response',
]
WINDOW_SHORT = {
    'full_trial': 'Full', 'init': 'Init', 'delay': 'Delay',
    'init_delay': 'I+D', 'response': 'Resp', 'delay_response': 'D+R',
}
WINDOW_FIG_SUFFIX = {
    'full_trial': 'full', 'init': 'init', 'delay': 'delay',
    'init_delay': 'init+delay', 'response': 'response',
    'delay_response': 'response+delay',
}
WINDOW_COLORS = {
    'full_trial': '#2C3E50', 'init': '#3498DB', 'delay': '#E67E22',
    'init_delay': '#27AE60', 'response': '#E74C3C', 'delay_response': '#9B59B6',
}


def set_publication_style():
    plt.rcParams.update({
        'font.size': FONT['tick'],
        'axes.labelsize': FONT['label'],
        'axes.titlesize': FONT['title'],
        'xtick.labelsize': FONT['tick'],
        'ytick.labelsize': FONT['tick'],
        'legend.fontsize': FONT['legend'],
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })


# =====================================================================
#  Helpers
# =====================================================================

def _sem(arr, axis=None):
    """Standard error of the mean."""
    a = np.asarray(arr, dtype=float)
    n = a.size if axis is None else a.shape[axis]
    return np.std(a, axis=axis, ddof=0) / np.sqrt(max(n, 1))


def _std_to_sem(std_val, n_models):
    """Convert a stored standard deviation to s.e.m."""
    return std_val / np.sqrt(max(n_models, 1))


def _safe_plot(func, *args, label='plot', **kwargs):
    try:
        func(*args, **kwargs)
        print(f"  ✓ {label}")
    except Exception as e:
        print(f"  ✗ {label}: {e}")


def _legend_kw(loc='best', ncol=1, **extra):
    base = dict(loc=loc, ncol=ncol, fontsize=FONT['legend'],
                framealpha=0.80, edgecolor='#cccccc',
                handlelength=1.5, handleheight=0.8,
                handletextpad=0.4, borderpad=0.4,
                labelspacing=0.3, markerscale=0.9)
    base.update(extra)
    return base


def _square(ax):
    ax.set_box_aspect(1)


def _figsize(nrows, ncols, extra_w=0.0):
    return (ncols * (SUBPLOT_SIZE + 0.6) + extra_w,
            nrows * (SUBPLOT_SIZE + 0.7))


def _draw_epoch_markers(ax, init_steps, delay_steps, total_length):
    resp_start = init_steps + delay_steps
    ax.axvspan(0, init_steps, alpha=0.06, color=ZONE['init_bg'])
    ax.axvline(init_steps, color=ZONE['edge'], ls=':', lw=0.8, alpha=0.5)
    if delay_steps > 0:
        ax.axvspan(init_steps, resp_start, alpha=0.06, color=ZONE['delay_bg'])
        ax.axvline(resp_start, color=ZONE['edge'], ls=':', lw=0.8, alpha=0.5)


def _draw_sig_star(ax, x_pos, y_pos, text, color='black', fontsize=None):
    if not text or text == 'n.s.':
        return
    fs = fontsize if fontsize is not None else FONT['annot'] + 2
    ax.text(x_pos, y_pos, text, ha='center', va='bottom',
            fontsize=fs, fontweight='bold', color=color)


def _aggregate_transition_data(data_dict):
    """Aggregate TransitionData objects from multiple seeds."""
    first = list(data_dict.values())[0]

    agg = TransitionData()
    agg.init_steps = first.init_steps
    agg.delay_steps = first.delay_steps
    agg.response_steps = first.response_steps
    for data in data_dict.values():
        agg.n2w_trajectories.extend(data.n2w_trajectories)
        agg.w2n_trajectories.extend(data.w2n_trajectories)
        agg.nrem_only_trajectories.extend(data.nrem_only_trajectories)
        agg.wake_only_trajectories.extend(data.wake_only_trajectories)
    return agg


def _unify_multi_seed_data(intact_data, ablation_data=None):
    """Unify single-seed or multi-seed TransitionData into a common format.

    Returns (intact_agg, ablation_agg, init_steps, delay_steps,
             response_steps, subtitle_suffix)
    """
    if isinstance(intact_data, dict):
        first = list(intact_data.values())[0]
        if hasattr(first, 'init_steps'):
            n_models = len(intact_data)
            agg_intact = _aggregate_transition_data(intact_data)
            agg_ablation = None
            if ablation_data is not None:
                cond_names = list(list(ablation_data.values())[0].keys())
                agg_ablation = {}
                for cond in cond_names:
                    cond_dict = {
                        seed: ablation_data[seed][cond]
                        for seed in ablation_data
                        if cond in ablation_data[seed]
                    }
                    if cond_dict:
                        agg_ablation[cond] = _aggregate_transition_data(cond_dict)
            return (agg_intact, agg_ablation,
                    first.init_steps, first.delay_steps,
                    first.response_steps, f' (n={n_models} models)')

    init_s = intact_data.init_steps
    delay_s = intact_data.delay_steps
    resp_s = intact_data.response_steps
    return intact_data, ablation_data, init_s, delay_s, resp_s, ''


# ── Model-level SEM helpers ────────────────────────────────────

def _get_seed_data_for_cond(raw_intact, raw_ablation, cond_key):
    """Return {seed: TransitionData} for one condition.

    Parameters
    ----------
    raw_intact  : dict {seed: TransitionData}  (multi-seed) or single
    raw_ablation: dict {seed: {cond: TransitionData}}  (multi-seed) or single
    cond_key    : 'intact' or an ablation condition name

    Returns
    -------
    dict {seed: TransitionData}   (may be empty)
    """
    if cond_key == 'intact':
        if isinstance(raw_intact, dict):
            return raw_intact
        return {}
    if not isinstance(raw_ablation, dict):
        return {}
    out = {}
    for seed, val in raw_ablation.items():
        if isinstance(val, dict) and cond_key in val:
            out[seed] = val[cond_key]
    return out


def _model_level_sem(per_model_vals):
    """Given a list of 1-D arrays (one per model), return SEM across models.

    Returns None if fewer than 2 models.
    """
    if len(per_model_vals) < 2:
        return None
    # Truncate to shortest common length (safety)
    min_len = min(len(v) for v in per_model_vals)
    stacked = np.array([v[:min_len] for v in per_model_vals])  # (n_models, T)
    return np.std(stacked, axis=0, ddof=0) / np.sqrt(len(per_model_vals))


def _is_multi_seed(data):
    """Check whether data is a multi-seed dict of TransitionData."""
    if not isinstance(data, dict) or len(data) < 2:
        return False
    first = next(iter(data.values()))
    return hasattr(first, 'init_steps')


# ─────────────────────────────────────────────────────────────────────


def _infer_n_models(aggregate_dict):
    """Infer n_models from an aggregate results dict."""
    if not isinstance(aggregate_dict, dict):
        return 1
    n = aggregate_dict.get('n_models')
    if n is not None and n > 0:
        return int(n)
    return 1


def _extract_metrics_from_results(results, trial_types, conditions):
    """Extract accuracy / wake / nrem means & s.e.m.

    Returns (n_models, acc_mean, acc_sem, wake_mean, nrem_mean)
    """
    if isinstance(results, AblationResult):
        summary = results.summary
        n_models = 1
        acc_mean = {t: {c: summary.get(t, {}).get(c, {}).get('accuracy', 0)
                        for c in conditions} for t in trial_types}
        acc_sem = {t: {c: 0.0 for c in conditions} for t in trial_types}
        wake_mean = {t: {c: summary.get(t, {}).get(c, {}).get('wake_rate', 0)
                         for c in conditions} for t in trial_types}
        nrem_mean = {t: {c: summary.get(t, {}).get(c, {}).get('nrem_rate', 0)
                         for c in conditions} for t in trial_types}

    elif 'individual' in results:
        individual = results['individual']
        n_models = len(individual)
        acc_lists  = {t: {c: [] for c in conditions} for t in trial_types}
        wake_lists = {t: {c: [] for c in conditions} for t in trial_types}
        nrem_lists = {t: {c: [] for c in conditions} for t in trial_types}

        for seed, res in individual.items():
            if hasattr(res, 'ablation_result'):
                summary = res.ablation_result.summary
            elif hasattr(res, 'summary'):
                summary = res.summary
            else:
                continue
            for trial_type in trial_types:
                for cond in conditions:
                    if trial_type in summary and cond in summary[trial_type]:
                        entry = summary[trial_type][cond]
                        acc_lists[trial_type][cond].append(
                            entry.get('accuracy', 0))
                        wake_lists[trial_type][cond].append(
                            entry.get('wake_rate', 0))
                        nrem_lists[trial_type][cond].append(
                            entry.get('nrem_rate', 0))

        acc_mean  = {t: {c: np.mean(acc_lists[t][c]) if acc_lists[t][c] else 0
                         for c in conditions} for t in trial_types}
        acc_sem   = {t: {c: _sem(acc_lists[t][c]) if acc_lists[t][c] else 0
                         for c in conditions} for t in trial_types}
        wake_mean = {t: {c: np.mean(wake_lists[t][c]) if wake_lists[t][c] else 0
                         for c in conditions} for t in trial_types}
        nrem_mean = {t: {c: np.mean(nrem_lists[t][c]) if nrem_lists[t][c] else 0
                         for c in conditions} for t in trial_types}

    else:
        agg = results.get('aggregate', results)
        n_models = _infer_n_models(agg)

        acc_mean = {t: {c: agg.get(t, {}).get(c, {}).get('accuracy', 0)
                        for c in conditions} for t in trial_types}
        acc_sem  = {t: {c: _std_to_sem(
                            agg.get(t, {}).get(c, {}).get('accuracy_std', 0),
                            n_models)
                        for c in conditions} for t in trial_types}
        wake_mean = {}
        nrem_mean = {}
        for t in trial_types:
            wake_mean[t] = {}
            nrem_mean[t] = {}
            for c in conditions:
                entry = agg.get(t, {}).get(c, {})
                wake_mean[t][c] = entry.get('wake_rate', 0)
                nrem_mean[t][c] = entry.get('nrem_rate', 0)

    return (n_models, acc_mean, acc_sem, wake_mean, nrem_mean)


# =====================================================================
#  1. Ablation Summary (3 × N_cond)
# =====================================================================

def plot_ablation_summary(results, save_path=None, significance_markers=None):
    """Accuracy / state-distribution bars per condition.
    Error bars: s.e.m. across models.
    """
    set_publication_style()
    conditions = CONDITIONS
    cond_titles = COND_TITLES
    trial_types = TRIAL_TYPES
    n_conds = len(conditions)

    (n_models, acc_mean, acc_sem,
     wake_mean, nrem_mean) = _extract_metrics_from_results(
        results, trial_types, conditions)

    sig_acc = {}
    if significance_markers is not None:
        if 'accuracy' in significance_markers:
            sig_acc = significance_markers['accuracy']
        else:
            sig_acc = significance_markers

    fig, axes = plt.subplots(2, n_conds, figsize=_figsize(2, n_conds))
    x_pos = np.arange(4)
    bar_width = 0.6
    intact_acc = [acc_mean[t]['intact'] for t in trial_types]

    # ── Row 0: Accuracy ──
    for col_idx, (cond, title) in enumerate(zip(conditions, cond_titles)):
        ax = axes[0, col_idx]
        _square(ax)
        means  = [acc_mean[t][cond] for t in trial_types]
        errors = [acc_sem[t][cond]  for t in trial_types]
        bar_colors = [
            PERF_COLORS['poor'] if baseline - val > 0.4
            else PERF_COLORS['moderate'] if baseline - val > 0.35
            else PERF_COLORS['good']
            for val, baseline in zip(means, intact_acc)
        ]
        bars = ax.bar(x_pos, means, bar_width, color=bar_colors,
                      edgecolor=ZONE['edge'], linewidth=0.8,
                      yerr=errors, capsize=3,
                      error_kw={'linewidth': 1.0, 'capthick': 1.0})
        ax.axhline(1 / 2, color=ZONE['chance'], ls='--', lw=1, alpha=0.7)

        for i, (bar, val, err, baseline) in enumerate(
                zip(bars, means, errors, intact_acc)):
            y_top = max(bar.get_height() + err + 0.02, 0.05)
            txt = f'{val:.0%}±{err:.0%}' if err > 0.005 else f'{val:.0%}'
            ax.text(bar.get_x() + bar.get_width() / 2, y_top, txt,
                    ha='center', va='bottom', fontsize=FONT['bar_text'])
            drop = baseline - val
            if drop > 0.4:
                ax.text(i, y_top + 0.07, f'↓{drop:.0%}', ha='center',
                        fontsize=FONT['annot'], color=COLORS['ablated'])
            if sig_acc and cond != 'intact':
                star = sig_acc.get(trial_types[i], {}).get(cond, '')
                if star and star != 'n.s.':
                    star_y = y_top + (0.12 if drop > 0.4 else 0.07)
                    _draw_sig_star(ax, i, star_y, star,
                                   color=COLORS['ablated'])

        ax.set_xticks(x_pos)
        ax.set_xticklabels(TRIAL_LABELS_SHORT, fontsize=FONT['tick'])
        ax.set_title(title, fontsize=FONT['title'], fontweight='bold')
        ax.set_ylim(0, 1.25)
        ax.yaxis.set_major_formatter(PercentFormatter(1, 0))
        ax.set_ylabel('Accuracy', fontsize=FONT['label'])

    # ── Row 1: State Distribution (Wake + NREM only) ──
    for col_idx, (cond, title) in enumerate(zip(conditions, cond_titles)):
        ax = axes[1, col_idx]
        _square(ax)
        wake_vals = [wake_mean[t][cond] for t in trial_types]
        nrem_vals = [nrem_mean[t][cond] for t in trial_types]

        ax.bar(x_pos, wake_vals, bar_width, label='Wake',
               color=STATE_COLORS['wake'],
               edgecolor=ZONE['edge'], linewidth=0.8)
        ax.bar(x_pos, nrem_vals, bar_width, bottom=wake_vals,
               label='NREM', color=STATE_COLORS['nrem'],
               edgecolor=ZONE['edge'], linewidth=0.8)
        ax.axhline(0.5, color=ZONE['chance'], ls='--', lw=1, alpha=0.5)

        for i, (w, n) in enumerate(zip(wake_vals, nrem_vals)):
            if w > 0.08:
                ax.text(i, w / 2, f'{w:.0%}', ha='center', va='center',
                        fontsize=FONT['bar_text'], color='white')
            if n > 0.08:
                ax.text(i, w + n / 2, f'{n:.0%}', ha='center', va='center',
                        fontsize=FONT['bar_text'], color='white')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(TRIAL_LABELS_SHORT, fontsize=FONT['tick'])
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(PercentFormatter(1, 0))
        ax.set_ylabel('State Distribution', fontsize=FONT['label'])
        ax.legend(**_legend_kw('upper left'))

    title_text = 'Path Ablation Summary'
    if n_models > 1:
        title_text += f'  (n = {n_models} models, mean ± s.e.m.)'
    plt.suptitle(title_text, fontsize=FONT['suptitle'],
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close()


# =====================================================================
#  2. Output Evolution (3 × N_cond)
#     FIX: SEM computed across *models* (not individual trials)
# =====================================================================

def plot_output_evolution(intact_data, ablation_data, save_path=None):
    """Output channel traces.  Shaded bands = s.e.m. across models.

    When multi-seed data is available, SEM is computed over per-model
    mean traces (capturing model-to-model variability), NOT over
    individual trials (which gives near-zero SEM due to large n).
    """
    set_publication_style()

    # ── Preserve raw seed-level data for model-level SEM ──
    raw_intact = intact_data
    raw_ablation = ablation_data
    is_multi = _is_multi_seed(intact_data)

    (data_intact, data_ablation, init_steps, delay_steps,
     response_steps, subtitle) = _unify_multi_seed_data(
        intact_data, ablation_data)
    if data_ablation is None:
        print("  SKIP output_evolution: no ablation data")
        return

    resp_start = init_steps + delay_steps
    total_length = init_steps + delay_steps + response_steps
    time_axis = np.arange(total_length)

    trajectory_configs = [
        ('n2w_trajectories', COLORS['n2w'], 'N→W'),
        ('w2n_trajectories', COLORS['w2n'], 'W→N'),
        ('nrem_only_trajectories', COLORS['nrem'], 'N Only'),
        ('wake_only_trajectories', COLORS['wake'], 'W Only'),
    ]

    # Build condition list: (title, aggregated_data, condition_key)
    all_conditions = [('Intact', data_intact, 'intact')]
    for cond, title in zip(CONDITIONS[1:], COND_TITLES[1:]):
        all_conditions.append((title, data_ablation.get(cond), cond))
    n_conds = len(all_conditions)
    channel_labels = ['Wake Channel', 'NREM Channel', 'Max Channel']

    fig, axes = plt.subplots(3, n_conds, figsize=_figsize(3, n_conds))
    for col_idx, (title, data, cond_key) in enumerate(all_conditions):
        if data is None:
            for row_idx in range(3):
                axes[row_idx, col_idx].set_title(
                    f'{title}\n(No Data)', fontsize=FONT['title'])
            continue
        for row_idx in range(3):
            ax = axes[row_idx, col_idx]
            _square(ax)
            for attr_name, color, label in trajectory_configs:
                trajs = getattr(data, attr_name, [])
                if not trajs:
                    continue
                # Check trajectory format (dict vs raw array)
                is_dict_fmt = isinstance(trajs[0], dict)
                if is_dict_fmt and 'output_trajectory' not in trajs[0]:
                    continue

                # ── Extract outputs from aggregated data ──
                if row_idx < 2:
                    if is_dict_fmt:
                        outputs = np.array([
                            td['output_trajectory'][:total_length, row_idx]
                            for td in trajs])
                    else:
                        continue  # raw arrays have no output_trajectory
                else:
                    if is_dict_fmt:
                        all_outputs = np.array([
                            td['output_trajectory'][:total_length]
                            for td in trajs])
                    else:
                        continue
                    wake_final = all_outputs[:, -1, 0].mean()
                    nrem_final = all_outputs[:, -1, 1].mean()
                    channel_idx = 0 if wake_final >= nrem_final else 1
                    outputs = all_outputs[:, :, channel_idx]

                mean_trace = outputs.mean(0)

                # ── Model-level SEM (preferred) ──
                sem_trace = None
                if is_multi:
                    seed_datas = _get_seed_data_for_cond(
                        raw_intact, raw_ablation, cond_key)
                    per_model = []
                    for seed, sdata in seed_datas.items():
                        strajs = getattr(sdata, attr_name, [])
                        if not strajs:
                            continue
                        s_dict_fmt = isinstance(strajs[0], dict)
                        if s_dict_fmt:
                            valid = [td for td in strajs
                                     if 'output_trajectory' in td]
                        else:
                            valid = []
                        if not valid:
                            continue
                        if row_idx < 2:
                            seed_out = np.array([
                                td['output_trajectory'][:total_length, row_idx]
                                for td in valid])
                        else:
                            seed_all = np.array([
                                td['output_trajectory'][:total_length]
                                for td in valid])
                            seed_out = seed_all[:, :, channel_idx]
                        per_model.append(seed_out.mean(axis=0))
                    sem_trace = _model_level_sem(per_model)

                # ── Fallback: trial-level SEM ──
                if sem_trace is None:
                    sem_trace = _sem(outputs, 0)

                ax.plot(time_axis[:len(mean_trace)], mean_trace,
                        color=color, lw=2, label=label)
                ax.fill_between(time_axis[:len(mean_trace)],
                                mean_trace - sem_trace,
                                mean_trace + sem_trace,
                                color=color, alpha=0.18)
            if row_idx < 2:
                ax.axhline(0.5, color=ZONE['chance'],
                           ls='--', lw=0.8, alpha=0.4)
            _draw_epoch_markers(ax, init_steps, delay_steps, total_length)
            ax.set_xlim(0, total_length - 1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_ylabel(channel_labels[row_idx])
            ax.set_xlabel('Time (bins)')
            if row_idx == 0:
                ax.set_title(title, fontsize=FONT['title'], fontweight='bold')
            ax.legend(**_legend_kw('upper right' if row_idx == 2 else 'best'))

    sem_label = 'across-model s.e.m.' if is_multi else 's.e.m.'
    plt.suptitle(
        f'Output Evolution{subtitle}  (shaded = {sem_label})',
        fontsize=FONT['suptitle'], fontweight='bold', y=1.0)
    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close()


# =====================================================================
#  3. State Transition Dynamics (1 × N_cond)
#     FIX: SEM computed across *models* (not individual trials)
# =====================================================================

def _compute_state_axis(intact_data):
    """Compute normalised NREM→Wake rule axis and normalisation params."""
    nrem_endpoints = np.array([
        td['end_state'] for td in intact_data.nrem_only_trajectories])
    wake_endpoints = np.array([
        td['end_state'] for td in intact_data.wake_only_trajectories])
    rule_axis = wake_endpoints.mean(0) - nrem_endpoints.mean(0)
    rule_axis /= np.linalg.norm(rule_axis)
    nrem_proj = float(nrem_endpoints.mean(0) @ rule_axis)
    wake_proj = float(wake_endpoints.mean(0) @ rule_axis)
    midpoint = (nrem_proj + wake_proj) / 2
    half_range = (wake_proj - nrem_proj) / 2
    return rule_axis, midpoint, half_range


def plot_state_transition_dynamics(intact_data, ablation_data, save_path=None):
    """Rule-axis projection.  Shaded bands = s.e.m. across models.

    When multi-seed data is available, SEM is computed over per-model
    mean projections, NOT over individual trials.
    """
    set_publication_style()

    # ── Preserve raw seed-level data ──
    raw_intact = intact_data
    raw_ablation = ablation_data
    is_multi = _is_multi_seed(intact_data)

    (data_intact, data_ablation, init_steps, delay_steps,
     response_steps, subtitle) = _unify_multi_seed_data(
        intact_data, ablation_data)
    if data_ablation is None:
        print("  SKIP state_transition: no ablation data")
        return

    total_length = init_steps + delay_steps + response_steps
    time_axis = np.arange(total_length)

    # Rule axis defined from aggregated intact data (stable estimate)
    rule_axis, midpoint, half_range = _compute_state_axis(data_intact)

    def normalize_projection(projection):
        return (projection - midpoint) / half_range

    trajectory_configs = [
        ('n2w_trajectories', COLORS['n2w'], 'N→W'),
        ('w2n_trajectories', COLORS['w2n'], 'W→N'),
        ('nrem_only_trajectories', COLORS['nrem'], 'N Only'),
        ('wake_only_trajectories', COLORS['wake'], 'W Only'),
    ]

    all_conditions = [('Intact', data_intact, 'intact')]
    for cond, title in zip(CONDITIONS[1:], COND_TITLES[1:]):
        all_conditions.append((title, data_ablation.get(cond), cond))
    n_conds = len(all_conditions)

    fig, axes = plt.subplots(1, n_conds, figsize=_figsize(1, n_conds))
    for col_idx, (title, data, cond_key) in enumerate(all_conditions):
        ax = axes[col_idx]
        _square(ax)
        if data is None:
            ax.set_title(f'{title}\n(No Data)', fontsize=FONT['title'])
            continue
        ax.axhline(-1, color=COLORS['nrem'], ls='--', lw=1.5, alpha=0.6)
        ax.axhline(+1, color=COLORS['wake'], ls='--', lw=1.5, alpha=0.6)
        ax.axhline(0, color=ZONE['chance'], ls='-', lw=0.6, alpha=0.3)
        ax.axhspan(-3, 0, alpha=0.04, color=COLORS['nrem'])
        ax.axhspan(0, 3, alpha=0.04, color=COLORS['wake'])

        for attr_name, color, label in trajectory_configs:
            trajs = getattr(data, attr_name, [])
            if not trajs:
                continue
            is_dict_fmt = isinstance(trajs[0], dict)
            if is_dict_fmt:
                projections = [
                    normalize_projection(
                        td['trajectory'][:total_length] @ rule_axis)
                    for td in trajs if 'trajectory' in td]
            else:
                # raw ndarray: shape (T, hidden_dim)
                projections = [
                    normalize_projection(td[:total_length] @ rule_axis)
                    for td in trajs]
            if not projections:
                continue

            mean_proj = np.mean(projections, 0)

            # ── Model-level SEM (preferred) ──
            sem_proj = None
            if is_multi:
                seed_datas = _get_seed_data_for_cond(
                    raw_intact, raw_ablation, cond_key)
                per_model = []
                for seed, sdata in seed_datas.items():
                    strajs = getattr(sdata, attr_name, [])
                    if not strajs:
                        continue
                    s_dict_fmt = isinstance(strajs[0], dict)
                    if s_dict_fmt:
                        seed_projs = [
                            normalize_projection(
                                td['trajectory'][:total_length] @ rule_axis)
                            for td in strajs if 'trajectory' in td]
                    else:
                        seed_projs = [
                            normalize_projection(
                                td[:total_length] @ rule_axis)
                            for td in strajs]
                    if seed_projs:
                        per_model.append(np.mean(seed_projs, axis=0))
                sem_proj = _model_level_sem(per_model)

            # ── Fallback: trial-level SEM ──
            if sem_proj is None:
                sem_proj = _sem(projections, 0)

            T = len(mean_proj)
            ax.plot(time_axis[:T], mean_proj, color=color, lw=2, label=label)
            ax.fill_between(time_axis[:T],
                            mean_proj - sem_proj,
                            mean_proj + sem_proj,
                            color=color, alpha=0.4)

        _draw_epoch_markers(ax, init_steps, delay_steps, total_length)
        ax.set_xlim(0, total_length - 1)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlabel('Time (bins)')
        ax.set_ylabel('State')
        ax.set_title(title, fontsize=FONT['title'], fontweight='bold')
        ax.legend(**_legend_kw('upper right'))
        ax.text(total_length - 1, 1.15, 'Wake', fontsize=FONT['annot'],
                color=COLORS['wake'], ha='right', fontstyle='italic')
        ax.text(total_length - 1, -1.15, 'NREM', fontsize=FONT['annot'],
                color=COLORS['nrem'], ha='right', fontstyle='italic')

    sem_label = 'across-model s.e.m.' if is_multi else 's.e.m.'
    plt.suptitle(
        f'State Transition Dynamics{subtitle}  (shaded = {sem_label})',
        fontsize=FONT['suptitle'], fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close()

# =====================================================================
#  5. Neuron Characterization
# =====================================================================

# =====================================================================
#  5. Neuron Characterization — Per-seed: Pie + Full-Trial SqI & PE
# =====================================================================



def plot_neuron_characterization_v1(neuron_chars, save_path=None):
    """Functional group sizes, peak-time histograms, CDF."""
    set_publication_style()
    seeds = sorted(neuron_chars.keys())
    ns = len(seeds)
    first = neuron_chars[seeds[0]]
    N = first.n_neurons
    tw = first.temporal_windows
    init_steps = tw['init'][1]
    delay_end = tw['analysis'][1]
    delay_steps = delay_end - init_steps
    total_steps = init_steps + delay_steps

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))

    # ── A. Group Sizes ──
    ax = axes[0, 0]
    gnames = ['N→W\ntrans', 'W→N\ntrans', 'NREM\nmaint', 'Wake\nmaint', 'Unclass.']
    gcounts = {g: [] for g in gnames}
    for s in seeds:
        c = neuron_chars[s]
        gcounts[gnames[0]].append(len(c.n2w_ids))
        gcounts[gnames[1]].append(len(c.w2n_ids))
        gcounts[gnames[2]].append(len(c.nrem_maint_ids))
        gcounts[gnames[3]].append(len(c.wake_maint_ids))
        gcounts[gnames[4]].append(len(c.unclassified_ids))
    x_pos = np.arange(len(gnames))
    means = [np.mean(gcounts[g]) for g in gnames]
    sems = [np.std(gcounts[g]) / np.sqrt(max(ns, 1)) for g in gnames]
    palette = [COLORS['n2w'], COLORS['w2n'], COLORS['nrem'], COLORS['wake'], '#ABB2B9']
    bars = ax.bar(x_pos, means, 0.6, yerr=sems, capsize=3, color=palette,
                  edgecolor=ZONE['edge'], linewidth=0.8,
                  error_kw={'linewidth': 1.0, 'capthick': 1.0})
    if ns > 1:
        rng = np.random.RandomState(42)
        for gi, g in enumerate(gnames):
            jitter = rng.uniform(-0.12, 0.12, size=ns)
            ax.scatter(np.full(ns, gi) + jitter, gcounts[g],
                       c='black', s=14, alpha=0.35, zorder=5, edgecolors='none')
    for bar, m, se in zip(bars, means, sems):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + se + 0.8,
                f'{int(round(m))}', ha='center', va='bottom', fontsize=FONT['bar_text'])
    ax.set_xticks(x_pos); ax.set_xticklabels(gnames, fontsize=FONT['tick'] - 1)
    ax.set_ylabel('Neuron count'); ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('A. Functional Group Sizes', fontsize=FONT['title'], fontweight='bold')

    # ── Timeline helper ──
    def _add_timeline(ax):
        ax.axvspan(0, init_steps, alpha=0.07, color='#3498DB')
        ax.axvspan(init_steps, total_steps, alpha=0.07, color='#E67E22')
        ax.axvline(init_steps, color='#7F8C8D', ls='--', lw=1.0, alpha=0.6)
        ax.text(init_steps / 2, 0.97, 'Init', transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=FONT['annot'] - 1,
                color='#2980B9', fontstyle='italic')
        ax.text(init_steps + delay_steps / 2, 0.97, 'Delay',
                transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=FONT['annot'] - 1, color='#D35400', fontstyle='italic')

    def _plot_peak_hist(ax, dkey, color, dlabel, panel):
        _add_timeline(ax)
        pt_list, pe_vals, sqi_vals = [], [], []
        for s in seeds:
            c = neuron_chars[s]
            pt_list.append(getattr(c, f'{dkey}_peak_times'))
            pe_vals.append(getattr(c, f'{dkey}_peak_entropy'))
            sqi_vals.append(getattr(c, f'{dkey}_seq_index'))
        all_pt = np.concatenate([p for p in pt_list if len(p) > 0]) if pt_list else np.array([])
        if len(all_pt) > 0:
            bins = np.linspace(0, total_steps, min(total_steps, 10) + 1)
            ax.hist(all_pt, bins=bins, color=color, edgecolor='white', linewidth=0.5, alpha=0.85)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        pe_m, sqi_m = np.mean(pe_vals), np.mean(sqi_vals)
        if ns > 1:
            ann = f'PE = {pe_m:.2f}±{np.std(pe_vals)/np.sqrt(ns):.2f}\nSqI = {sqi_m:.2f}'
        else:
            ann = f'PE = {pe_m:.2f}\nSqI = {sqi_m:.2f}'
        ax.text(0.97, 0.88, ann, transform=ax.transAxes, ha='right', va='top',
                fontsize=FONT['annot'],
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#BDC3C7', alpha=0.9))
        ax.set_xlabel('Time step'); ax.set_ylabel('Neuron count'); ax.set_xlim(0, total_steps)
        ax.set_title(f'{panel}. {dlabel} Sequence Peak Times',
                     fontsize=FONT['title'], fontweight='bold')

    # ── B. N→W Peak-Time ──
    _plot_peak_hist(axes[0, 1], 'n2w', COLORS['n2w'], 'N→W', 'B')

    # ── C. W→N Peak-Time ──
    _plot_peak_hist(axes[1, 0], 'w2n', COLORS['w2n'], 'W→N', 'C')

    # ── D. CDF ──
    ax = axes[1, 1]; _add_timeline(ax)
    for si, s in enumerate(seeds):
        c = neuron_chars[s]
        a_val = max(0.3, 0.85 / max(ns, 1))
        for pt, color, lab in [(c.n2w_peak_times, COLORS['n2w'], 'N→W'),
                                (c.w2n_peak_times, COLORS['w2n'], 'W→N')]:
            if len(pt) > 0:
                sorted_pt = np.sort(pt); n_pts = len(sorted_pt)
                x_cdf = np.concatenate([[0], sorted_pt, [total_steps]])
                y_cdf = np.concatenate([[0], np.arange(1, n_pts + 1) / n_pts, [1.0]])
                ax.step(x_cdf, y_cdf, where='post', color=color, alpha=a_val, lw=1.5,
                        label=lab if si == 0 else None)
    ax.plot([0, total_steps], [0, 1], color='grey', ls=':', lw=1.0, alpha=0.5, label='Uniform')
    ax.set_xlim(0, total_steps)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel('Peak time step')
    ax.set_ylabel('Cumulative fraction')
    ax.legend(**_legend_kw('upper left'))
    ax.set_title('D. Sequence Structure (CDF)', fontsize=FONT['title'], fontweight='bold')

    nl = f'n = {ns} models' if ns > 1 else f'seed = {seeds[0]}'
    plt.suptitle(f'Neuron Functional Characterization  ({nl}, N = {N})',
                 fontsize=FONT['suptitle'], fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close()



def plot_neuron_characterization(neuron_chars, save_path=None,
                                 intact_data=None):
    """Per-seed neuron characterization (3 figures × N seeds).

    Sequence metrics (Sequentiality Index, Peak Entropy) are recomputed
    from **full-trial** intact trajectories (time 0 → T), NOT the
    init+delay window used for neuron classification.

    Figures saved to ``Figure_neuron/``:
      1. Pie chart  — 4 functional-group proportions
      2. Bar chart  — Sequentiality Index for all 4 groups
      3. Bar chart  — Peak Entropy for all 4 groups
    """
    set_publication_style()

    # ── Resolve output directory ──
    if save_path is not None:
        parent = os.path.dirname(save_path) or '.'
        fig_root = os.path.dirname(parent) or '.'
        neuron_fig_dir = os.path.join(fig_root, 'Figure_neuron')
    else:
        neuron_fig_dir = 'Figure_neuron'
    os.makedirs(neuron_fig_dir, exist_ok=True)

    seeds = sorted(neuron_chars.keys())

    # (neuron-id attribute, trajectory attribute, colour, label)
    GROUP_CFG = [
        ('n2w_ids',        'n2w_trajectories',       COLORS['n2w'],  'N→W'),
        ('w2n_ids',        'w2n_trajectories',       COLORS['w2n'],  'W→N'),
        ('nrem_maint_ids', 'nrem_only_trajectories', COLORS['nrem'], 'N Only'),
        ('wake_maint_ids', 'wake_only_trajectories', COLORS['wake'], 'W Only'),
    ]

    for seed in seeds:
        char = neuron_chars[seed]

        # ── Resolve intact trajectory data for this seed ──
        seed_tdata = None
        if intact_data is not None:
            if isinstance(intact_data, dict) and seed in intact_data:
                seed_tdata = intact_data[seed]
            elif hasattr(intact_data, 'n2w_trajectories'):
                seed_tdata = intact_data          # single TransitionData

        # ── Compute full-trial SqI / PE for all 4 groups ──
        sqi_vals, pe_vals, ts_vals = [], [], []
        n_neurons_per_group = []
        bar_labels, bar_colors = [], []

        for ids_attr, traj_attr, color, label in GROUP_CFG:
            neuron_ids = getattr(char, ids_attr)
            bar_labels.append(label)
            bar_colors.append(color)
            n_neurons_per_group.append(len(neuron_ids))

            if len(neuron_ids) == 0 or seed_tdata is None:
                sqi_vals.append(0.0)
                pe_vals.append(0.0)
                ts_vals.append(0.0)
                continue

            trajs = getattr(seed_tdata, traj_attr, [])
            if not trajs:
                sqi_vals.append(0.0)
                pe_vals.append(0.0)
                ts_vals.append(0.0)
                continue

            # Extract hidden-state arrays  (T, hidden_dim)
            if isinstance(trajs[0], dict):
                hidden_list = [td['trajectory'] for td in trajs
                               if 'trajectory' in td]
            else:
                hidden_list = list(trajs)

            if not hidden_list:
                sqi_vals.append(0.0)
                pe_vals.append(0.0)
                ts_vals.append(0.0)
                continue

            T = hidden_list[0].shape[0]-10           # Init+Delay+Response
            # Mean activity across trials: (hidden_dim, T)
            mean_act = np.mean(
                np.stack([h[10:].T for h in hidden_list]), axis=0)
            sub_act = mean_act[neuron_ids]        # (n_group, T)
            peak_times = sub_act.argmax(axis=1)   # (n_group,)

            sqi, pe, ts = _sequentiality_index(peak_times, sub_act, T)
            sqi_vals.append(sqi)
            pe_vals.append(pe)
            ts_vals.append(ts)

        has_metrics = any(v > 0 for v in sqi_vals) or any(v > 0 for v in pe_vals)

        # ════════════════════════════════════════════════════════
        #  Figure 1 — Pie Chart
        # ════════════════════════════════════════════════════════
        # fig1, ax1 = plt.subplots(figsize=(6.5, 6.5))

        # pie_names = ['N→W Trans', 'W→N Trans', 'NREM Maint', 'Wake Maint']
        # pie_sizes = [
        #     len(char.n2w_ids), len(char.w2n_ids),
        #     len(char.nrem_maint_ids), len(char.wake_maint_ids),
        # ]
        # pie_colors = [COLORS['n2w'], COLORS['w2n'],
        #               COLORS['nrem'], COLORS['wake']]

        # display = [(n, s, c) for n, s, c in
        #            zip(pie_names, pie_sizes, pie_colors) if s > 0]
        # if display:
        #     d_names, d_sizes, d_colors = zip(*display)

        #     def _autopct(pct, _sizes=d_sizes):
        #         count = int(round(pct / 100.0 * sum(_sizes)))
        #         return f'{pct:.1f}%\n({count})'

        #     wedges, texts, autotexts = ax1.pie(
        #         d_sizes, labels=d_names, colors=d_colors,
        #         autopct=_autopct, startangle=90, pctdistance=0.65,
        #         wedgeprops=dict(edgecolor='white', linewidth=2.0),
        #         textprops=dict(fontsize=FONT['label']),
        #     )
        #     for at in autotexts:
        #         at.set_fontsize(FONT['annot'])
        #         at.set_fontweight('bold')
        # else:
        #     ax1.text(0.5, 0.5, 'No classified neurons',
        #              transform=ax1.transAxes, ha='center', va='center',
        #              fontsize=FONT['title'], color='#888888')

        # n_classified = sum(pie_sizes)
        # n_unclass = len(char.unclassified_ids)
        # ax1.set_title(
        #     f'Functional Groups\n'
        #     f'seed={seed}   N={char.n_neurons}   '
        #     f'classified={n_classified}   unclassified={n_unclass}',
        #     fontsize=FONT['title'], fontweight='bold', pad=18)
        # fig1.tight_layout()
        # save_figure(fig1, os.path.join(neuron_fig_dir,
        #                              f'pie_chart_seed{seed}.png'))
        # plt.close(fig1)

        # if not has_metrics:
        #     print(f"    seed={seed}: no trajectory data — "
        #           f"SqI/PE figures skipped")
        #     continue

        # ════════════════════════════════════════════════════════
        #  Figure 2 — Sequentiality Index  (4 groups, full trial)
        # ════════════════════════════════════════════════════════
        fig2, ax2 = plt.subplots(figsize=(5.8, 4.8))
        x = np.arange(len(bar_labels))
        bars2 = ax2.bar(x, sqi_vals, 0.55, color=bar_colors,
                        edgecolor=ZONE['edge'], linewidth=0.8)

        y_max_sqi = max(max(sqi_vals), 0.01)
        for bar, val, nn in zip(bars2, sqi_vals, n_neurons_per_group):
            y_top = bar.get_height() + y_max_sqi * 0.03
            ax2.text(bar.get_x() + bar.get_width() / 2, y_top,
                     f'{val:.3f}', ha='center', va='bottom',
                     fontsize=FONT['bar_text'], fontweight='bold')
            # Show neuron count inside bar
            if bar.get_height() > y_max_sqi * 0.08:
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() * 0.5,
                         f'n={nn}', ha='center', va='center',
                         fontsize=FONT['annot'] - 1, color='white',
                         fontstyle='italic')

        ax2.set_xticks(x)
        ax2.set_xticklabels(bar_labels, fontsize=FONT['label'])
        ax2.set_ylabel('Sequentiality Index  (PE × TS)',
                       fontsize=FONT['label'])
        ax2.set_ylim(0, y_max_sqi * 1.35)

        # Annotate PE and TS breakdown
        info_lines = '  |  '.join(
            f'{lb}: PE={pe:.2f} TS={ts:.2f}'
            for lb, pe, ts in zip(bar_labels, pe_vals, ts_vals)
        )

        ax2.set_title(
            f'Sequentiality Index (seed={seed})',
            fontsize=FONT['title'], fontweight='bold')
        fig2.tight_layout(rect=[0, 0.05, 1, 1])
        save_figure(fig2, os.path.join(neuron_fig_dir,
                                     f'seq_index_seed{seed}.png'))
        plt.close(fig2)

        # # ════════════════════════════════════════════════════════
        # #  Figure 3 — Peak Entropy  (4 groups, full trial)
        # # ════════════════════════════════════════════════════════
        # fig3, ax3 = plt.subplots(figsize=(5.8, 4.8))
        # bars3 = ax3.bar(x, pe_vals, 0.55, color=bar_colors,
        #                 edgecolor=ZONE['edge'], linewidth=0.8)

        # y_max_pe = max(max(pe_vals), 0.01)
        # for bar, val, nn in zip(bars3, pe_vals, n_neurons_per_group):
        #     y_top = bar.get_height() + y_max_pe * 0.03
        #     ax3.text(bar.get_x() + bar.get_width() / 2, y_top,
        #              f'{val:.3f}', ha='center', va='bottom',
        #              fontsize=FONT['bar_text'], fontweight='bold')
        #     if bar.get_height() > y_max_pe * 0.08:
        #         ax3.text(bar.get_x() + bar.get_width() / 2,
        #                  bar.get_height() * 0.5,
        #                  f'n={nn}', ha='center', va='center',
        #                  fontsize=FONT['annot'] - 1, color='white',
        #                  fontstyle='italic')

        # ax3.set_xticks(x)
        # ax3.set_xticklabels(bar_labels, fontsize=FONT['label'])
        # ax3.set_ylabel('Peak Entropy  (normalised)',
        #                fontsize=FONT['label'])
        # ax3.set_ylim(0, min(y_max_pe * 1.35, 1.15))
        # ax3.axhline(1.0, color=ZONE['chance'], ls='--', lw=1,
        #             alpha=0.5, label='Uniform (max)')
        # ax3.legend(**_legend_kw('upper right'))
        # ax3.set_title(
        #     f'Peak Entropy (seed={seed})',
        #     fontsize=FONT['title'], fontweight='bold')
        # fig3.tight_layout()
        # save_figure(fig3, os.path.join(neuron_fig_dir,
        #                              f'peak_entropy_seed{seed}.png'))
        # plt.close(fig3)

    print(f"  Neuron figures → {neuron_fig_dir}  "
          f"({len(seeds)} seeds × 3 figures)")
    
# =====================================================================
#  6. Temporal Window Schematic
# =====================================================================

def plot_temporal_window_schematic(temporal_res, save_path=None):
    """Timeline diagram of trial structure with the 6 ablation windows."""
    if temporal_res is None:
        return
    set_publication_style()

    init_steps = temporal_res.init_steps
    delay_steps = temporal_res.delay_steps
    resp_steps = temporal_res.response_steps
    total = init_steps + delay_steps + resp_steps
    resp_start = init_steps + delay_steps

    fig, axes = plt.subplots(7, 1, figsize=(10, 6.5),
                              gridspec_kw={'height_ratios': [2] + [1] * 6})

    ax = axes[0]
    ax.barh(0, init_steps, left=0, height=0.6, color='#3498DB',
            alpha=0.7, edgecolor='white', linewidth=1.5)
    ax.barh(0, delay_steps, left=init_steps, height=0.6, color='#E67E22',
            alpha=0.7, edgecolor='white', linewidth=1.5)
    ax.barh(0, resp_steps, left=resp_start, height=0.6, color='#27AE60',
            alpha=0.7, edgecolor='white', linewidth=1.5)
    for phase_label, center_x in [('Init', init_steps / 2),
                                   ('Delay', init_steps + delay_steps / 2),
                                   ('Response', resp_start + resp_steps / 2)]:
        ax.text(center_x, 0, phase_label, ha='center', va='center',
                fontsize=FONT['label'], fontweight='bold', color='white')
    ax.set_xlim(0, total)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.axis('off')
    ax.set_title('Trial Structure & Temporal Ablation Windows',
                 fontsize=FONT['suptitle'], fontweight='bold', pad=10)

    window_definitions = [
        ('full_trial', 0, total),
        ('init', 0, init_steps),
        ('delay', init_steps, resp_start),
        ('init_delay', 0, resp_start),
        ('response', resp_start, total),
        ('delay_response', init_steps, total),
    ]
    for idx, (window_name, start, end) in enumerate(window_definitions):
        ax = axes[idx + 1]
        ax.barh(0, total, left=0, height=0.5, color='#F0F0F0',
                edgecolor='#CCCCCC', linewidth=0.5)
        ax.barh(0, end - start, left=start, height=0.5,
                color=WINDOW_COLORS[window_name], alpha=0.7,
                edgecolor='white', linewidth=1)
        ax.text(-0.5, 0, WINDOW_SHORT[window_name],
                ha='right', va='center', fontsize=FONT['tick'],
                fontweight='bold', color=WINDOW_COLORS[window_name],
                transform=ax.get_yaxis_transform())
        ax.text(total + 1, 0, f'steps {start}–{end}',
                ha='left', va='center', fontsize=FONT['tick'] - 1,
                color='#7F8C8D')
        ax.set_xlim(0, total)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.axis('off')

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close()


# =====================================================================
#  7. Temporal Heatmap
# =====================================================================

def plot_temporal_heatmap(temporal_res, metric='accuracy', save_path=None):
    """Heatmap: conditions (rows) × windows (cols), one subplot per trial type."""
    if temporal_res is None:
        return
    set_publication_style()

    windows = temporal_res.window_names
    if metric == 'accuracy':
        data_dict = temporal_res.accuracy_mean
        cmap = 'RdYlGn'
        vmin, vmax = 0, 1
        metric_label = 'Accuracy'

    ablation_conds = [c for c in CONDITIONS if c != 'intact']
    all_conds = ['intact'] + ablation_conds
    cond_labels = [COND_LABELS_MAP.get(c, c) for c in all_conds]
    window_labels = [WINDOW_SHORT[w] for w in windows]

    n_trial_types = len(TRIAL_TYPES)
    fig, axes = plt.subplots(
        1, n_trial_types,
        figsize=(5.5 * n_trial_types, 0.55 * len(all_conds) + 2))

    for trial_idx, trial_type in enumerate(TRIAL_TYPES):
        ax = axes[trial_idx]
        matrix = np.zeros((len(all_conds), len(windows)))
        for cond_idx, cond_name in enumerate(all_conds):
            for win_idx, window_name in enumerate(windows):
                matrix[cond_idx, win_idx] = data_dict.get(
                    window_name, {}).get(trial_type, {}).get(cond_name, 0)

        im = ax.imshow(matrix, aspect='auto', cmap=cmap,
                        vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_xticks(range(len(windows)))
        ax.set_xticklabels(window_labels, fontsize=FONT['tick'],
                           rotation=30, ha='right')
        ax.set_yticks(range(len(all_conds)))
        ax.set_yticklabels(
            cond_labels if trial_idx == 0 else [],
            fontsize=FONT['tick'])
        ax.set_title(TRIAL_LABELS[trial_type],
                     fontsize=FONT['title'], fontweight='bold')

        for cond_idx in range(len(all_conds)):
            for win_idx in range(len(windows)):
                val = matrix[cond_idx, win_idx]
                text_color = ('white'
                              if (val > 0.65 and metric == 'accuracy')
                              or (val > 0.5 and metric != 'accuracy')
                              else 'black')
                ax.text(win_idx, cond_idx, f'{val:.0%}',
                        ha='center', va='center',
                        fontsize=FONT['annot'] - 1, color=text_color)

        if trial_idx == n_trial_types - 1:
            _add_colorbar(fig, ax, im, metric_label)

    plt.suptitle(
        f'{metric_label} — Temporal Window × Condition  '
        f'(n={temporal_res.n_models} models)',
        fontsize=FONT['suptitle'], fontweight='bold', y=1.03)
    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close()


def _add_colorbar(fig, ax, mappable, label):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(label, fontsize=FONT['label'])
    cb.ax.yaxis.set_major_formatter(PercentFormatter(1, 0))


# =====================================================================
#  8. Layer 1 — 6-Window Comprehensive Contrast
# =====================================================================

def plot_layer1_6window_contrast(temporal_res, save_path=None):
    """8-panel comparison: 2 metrics × 4 trial types.

    Each panel: grouped bars (conditions × windows).
    Error bars = s.e.m. across models.
    """
    if temporal_res is None:
        return
    set_publication_style()

    windows = temporal_res.window_names
    all_conds = [('intact', 'Intact')] + [
        (c, COND_LABELS_MAP[c]) for c in CONDITIONS if c != 'intact']
    n_windows = len(windows)
    n_conds = len(all_conds)
    n_models = temporal_res.n_models
    sem_divisor = max(n_models, 1) ** 0.5

    metric_configs = [
        dict(key='accuracy',
             mean_data=temporal_res.accuracy_mean,
             std_data=temporal_res.accuracy_std,
             ylabel='Accuracy',
             chance_line=1 / 2),
    ]

    n_trial_types = len(TRIAL_TYPES)
    fig, axes = plt.subplots(
        len(metric_configs), n_trial_types,
        figsize=(7.0 * n_trial_types, 6.5 * len(metric_configs)),
        squeeze=False)
    bar_width = 0.11

    for row_idx, metric_cfg in enumerate(metric_configs):
        mean_dict = metric_cfg['mean_data'] or {}
        std_dict = metric_cfg['std_data']
        for col_idx, trial_type in enumerate(TRIAL_TYPES):
            ax = axes[row_idx, col_idx]
            for win_idx, window_name in enumerate(windows):
                values, errors = [], []
                for cond_key, _ in all_conds:
                    val = mean_dict.get(window_name, {}).get(
                        trial_type, {}).get(cond_key, 0)
                    values.append(val)
                    err = (std_dict.get(window_name, {}).get(
                               trial_type, {}).get(cond_key, 0) / sem_divisor
                           if std_dict else 0)
                    errors.append(err)

                x_pos = np.arange(n_conds)
                offset = (win_idx - (n_windows - 1) / 2) * bar_width
                show_label = (row_idx == 0 and col_idx == 0)
                ax.bar(x_pos + offset, values, bar_width, yerr=errors,
                       color=WINDOW_COLORS[window_name], alpha=0.85,
                       capsize=2, edgecolor='white', linewidth=0.5,
                       error_kw={'linewidth': 0.8},
                       label=WINDOW_SHORT[window_name] if show_label else '')

            ax.set_xticks(np.arange(n_conds))
            ax.set_xticklabels([label for _, label in all_conds],
                               rotation=30, ha='right', fontsize=8)
            ax.set_ylabel(metric_cfg['ylabel'], fontsize=10)
            ax.set_ylim(0, 1.18)
            ax.yaxis.set_major_formatter(PercentFormatter(1, 0))
            ax.grid(axis='y', alpha=0.15, zorder=0)
            if metric_cfg['chance_line'] is not None:
                ax.axhline(metric_cfg['chance_line'],
                           ls=':', color='gray', alpha=0.5)
            if row_idx == 0:
                ax.set_title(TRIAL_LABELS[trial_type],
                             fontsize=13, fontweight='bold', pad=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    valid_entries = [(h, l) for h, l in zip(handles, labels) if l]
    if valid_entries:
        fig.legend(*zip(*valid_entries), loc='upper center',
                   ncol=n_windows, fontsize=11,
                   bbox_to_anchor=(0.5, 1.01),
                   frameon=True, edgecolor='#ccc', fancybox=True)

    fig.suptitle(
        f'Layer 1 — 6-Window Ablation Comparison  '
        f'(n = {n_models} models, error bars = s.e.m.)',
        fontsize=16, fontweight='bold', y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, save_path)
    plt.close(fig)


# =====================================================================
#  Entry Points
# =====================================================================

def generate_per_window_figures(
    *,
    window_name: str,
    fig_dir: str,
    analysis_results: Dict,
    intact_data=None,
    ablation_data=None,
    neuron_chars=None,
    statistics=None,
):
    """Generate the full set of per-window figures for one temporal window."""
    os.makedirs(fig_dir, exist_ok=True)
    window_label = WINDOW_SHORT.get(window_name, window_name)
    has_trajectories = intact_data is not None and bool(intact_data)
    has_aggregate = analysis_results is not None

    if has_trajectories:
        n_seeds = len(intact_data) if isinstance(intact_data, dict) else 1
        print(f"  {window_label}: trajectories available ({n_seeds} seeds)")
    else:
        print(f"  {window_label}: NO trajectory data — "
              f"output_evolution/state_transition will be skipped")

    print(f"\n  ── Per-window figures: {window_label} ──")

    sig_markers = None
    if statistics is not None:
        sig_markers = _extract_significance_markers(statistics)

    if has_aggregate:
        _safe_plot(
            plot_ablation_summary, analysis_results,
            save_path=os.path.join(fig_dir, 'ablation_summary.png'),
            significance_markers=sig_markers,
            label=f'ablation_summary [{window_label}]')

    if has_trajectories:
        _safe_plot(
            plot_output_evolution, intact_data, ablation_data,
            save_path=os.path.join(fig_dir, 'output_evolution.png'),
            label=f'output_evolution [{window_label}]')

        _safe_plot(
            plot_state_transition_dynamics, intact_data, ablation_data,
            save_path=os.path.join(fig_dir, 'state_transition_dynamics.png'),
            label=f'state_transition [{window_label}]')
    else:
        print(f"  ⚠ Trajectory-dependent figures skipped [{window_label}]")
    
    if neuron_chars:
        _safe_plot(
            plot_neuron_characterization, neuron_chars,
            save_path=os.path.join(fig_dir, 'neuron_characterization.png'),
            intact_data=intact_data,
            label=f'neuron_characterization [{window_label}]')
        
    # 6. Neuron characterization (window-independent)
    if neuron_chars:
        _safe_plot(plot_neuron_characterization_v1, neuron_chars,
                   save_path=os.path.join(fig_dir, 'neuron_characterization.png'),
                   label=f'neuron_characterization [{window_label}]')



def generate_all_ablation_figures(
    config,
    analysis_results: Dict,
    statistics=None,
):
    """Main entry: generate all quantitative ablation figures."""
    from analysis_core import load_pickle

    individual = analysis_results.get('individual', {})
    aggregate = analysis_results.get('aggregate', {})
    temporal_res = analysis_results.get('temporal')
    td_dicts_all = analysis_results.get('temporal_trajectories', {})
    phase_a_intact = analysis_results.get('intact_data')
    phase_a_ablation = analysis_results.get('ablation_data')

    neuron_char_dict = {
        seed: res.neuron_characterization
        for seed, res in individual.items()
        if hasattr(res, 'neuron_characterization')
        and res.neuron_characterization is not None
    }
    all_seeds = sorted(individual.keys())
    fig_root = os.path.dirname(config.paths.figure_dir)

    temporal_dir = os.path.join(config.paths.data_dir, 'analysis', 'temporal')

    n_in_memory = sum(1 for s in (td_dicts_all or {}).values() if s)
    n_on_disk = 0
    traj_disk_dir = os.path.join(temporal_dir, 'trajectories')
    if os.path.isdir(traj_disk_dir):
        n_on_disk = len([f for f in os.listdir(traj_disk_dir)
                         if f.endswith('.pkl')])
    print(f"  Trajectory data: {n_in_memory} seeds in-memory, "
          f"{n_on_disk} files on disk, "
          f"phase_a={'yes' if phase_a_intact else 'no'}")

    # ═════════════════════════════════════════════
    #  A. Per-window figure sets
    # ═════════════════════════════════════════════
    for window_name in WINDOW_KEYS:
        suffix = WINDOW_FIG_SUFFIX[window_name]
        fig_dir = os.path.join(fig_root, f'figures_{suffix}')

        print(f"\n{'═' * 50}")
        print(f"  Window: {WINDOW_SHORT[window_name]} → figures_{suffix}/")
        print(f"{'═' * 50}")

        window_intact, window_ablation = _extract_per_window_trajectories(
            td_dicts_all, window_name, phase_a_intact, phase_a_ablation,
            temporal_dir=temporal_dir, all_seeds=all_seeds)

        window_analysis = _build_window_aggregate_dict(
            temporal_res, window_name, aggregate)

        generate_per_window_figures(
            window_name=window_name,
            fig_dir=fig_dir,
            analysis_results=window_analysis,
            intact_data=window_intact if window_intact else None,
            ablation_data=window_ablation if window_ablation else None,
            neuron_chars=neuron_char_dict if neuron_char_dict else None,
            statistics=statistics,
        )

    # ═════════════════════════════════════════════
    #  B. Figure inventory
    # ═════════════════════════════════════════════
    print(f"\n{'─' * 50}")
    print("  Figure inventory:")
    key_files = ['ablation_summary.png', 'output_evolution.png',
                 'state_transition_dynamics.png',
                 'gradual_ablation.png']
    for window_name in WINDOW_KEYS:
        directory = os.path.join(
            fig_root, f'figures_{WINDOW_FIG_SUFFIX[window_name]}')
        if os.path.isdir(directory):
            found = [f for f in key_files
                     if os.path.exists(os.path.join(directory, f))]
            n_total = len([f for f in os.listdir(directory)
                           if f.endswith('.png') and not f.startswith('.')])
            status = ' '.join('✓' if f in found else '✗' for f in key_files)
            print(f"    {WINDOW_FIG_SUFFIX[window_name]:>15}: "
                  f"{status}  ({n_total} PNGs)")
        else:
            print(f"    {WINDOW_FIG_SUFFIX[window_name]:>15}: "
                  f"directory missing")
    print(f"{'─' * 50}")
    print("\n All ablation figures generated.")

# ── Private helpers for entry points ──

def _extract_per_window_trajectories(
    td_dicts_all, window_name,
    phase_a_intact, phase_a_ablation,
    temporal_dir=None, all_seeds=None,
):
    """Extract (intact_data, ablation_data) for one temporal window."""
    intact_data, ablation_data = {}, {}

    if td_dicts_all:
        for seed, seed_data in td_dicts_all.items():
            window_data = seed_data.get(window_name, {})
            if 'intact' in window_data:
                intact_data[seed] = window_data['intact']
            ablation_conds = {
                k: v for k, v in window_data.items() if k != 'intact'}
            if ablation_conds:
                ablation_data[seed] = ablation_conds

    if not intact_data and temporal_dir is not None and all_seeds:
        from analysis_core import load_pickle

        for seed in all_seeds:
            path = os.path.join(
                temporal_dir, 'trajectories',
                f'seed{seed}_{window_name}.pkl')
            if os.path.exists(path):
                td_dict = load_pickle(path)
                if td_dict is not None:
                    if 'intact' in td_dict:
                        intact_data[seed] = td_dict['intact']
                    ablation_conds = {
                        k: v for k, v in td_dict.items() if k != 'intact'}
                    if ablation_conds:
                        ablation_data[seed] = ablation_conds

        if intact_data:
            print(f"    Loaded {window_name} trajectories from disk "
                  f"({len(intact_data)} seeds)")

    if window_name == 'full_trial':
        if not intact_data and phase_a_intact:
            intact_data = phase_a_intact
        if not ablation_data and phase_a_ablation:
            ablation_data = phase_a_ablation

    return intact_data, ablation_data


def _build_window_aggregate_dict(
    temporal_res, window_name, global_aggregate,
):
    """Build analysis_results-like dict for one temporal window."""
    if temporal_res is not None and hasattr(
            temporal_res, 'get_window_aggregate'):
        return {'aggregate': temporal_res.get_window_aggregate(window_name)}

    if window_name == 'full_trial' and global_aggregate:
        return {'aggregate': global_aggregate}

    return {'aggregate': {}}

def _extract_significance_markers(statistics):
    if statistics is None:
        return None

    sig_markers = statistics.get('significance_markers')
    if sig_markers and 'accuracy' in sig_markers:
        return {'accuracy': sig_markers['accuracy']}

    markers = {'accuracy': {}}
    tests = statistics.get('tests', {})
    abl_acc = tests.get('ablation_accuracy', {})

    def _p_to_stars(p_value):
        if p_value < 0.001:   return '***'
        elif p_value < 0.01:  return '**'
        elif p_value < 0.05:  return '*'
        return 'n.s.'

    for trial_type in TRIAL_TYPES:
        markers['accuracy'][trial_type] = {}
        for cond in CONDITIONS:
            if cond == 'intact':
                continue
            result = abl_acc.get(trial_type, {}).get(cond)
            if result is None:
                result = abl_acc.get(trial_type, {}).get(f'intact_vs_{cond}')
            if result is not None:
                p_val = result.get('p_corrected', result.get('p_value', 1.0))
                markers['accuracy'][trial_type][cond] = _p_to_stars(p_val)

    has_any = any(markers['accuracy'].get(t) for t in TRIAL_TYPES)
    return markers if has_any else None

