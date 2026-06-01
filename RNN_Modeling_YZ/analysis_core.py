import torch
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# =====================================================================
#  Constants
# =====================================================================

TRIAL_TYPES = ['nrem_to_wake', 'wake_to_nrem', 'nrem_only', 'wake_only']

CONDITIONS = [
    'intact', 'ablate_n2w', 'ablate_w2n'
]

EXPECTED_STATE_MAP = {
    'nrem_to_wake': 'wake',
    'wake_to_nrem': 'nrem',
    'nrem_only':    'nrem',
    'wake_only':    'wake',
}

TRIAL_SHORT = {
    'nrem_to_wake': 'N→W',
    'wake_to_nrem': 'W→N',
    'nrem_only':    'N Only',
    'wake_only':    'W Only',
}

CONDITION_SHORT = {
    'intact': 'Intact',
    'ablate_n2w': 'Cut N→W',
    'ablate_w2n': 'Cut W→N'
}

TRIAL_COLORS = {
    'nrem_to_wake': '#e74c3c',
    'wake_to_nrem': '#3498db',
    'nrem_only':    '#27ae60',
    'wake_only':    '#f39c12',
}


# =====================================================================
#  Data Classes (used by vis_ablation)
# =====================================================================

@dataclass
class TransitionData:
    """Per-condition trajectory data collected during ablation evaluation."""
    init_steps: int = 0
    delay_steps: int = 0
    response_steps: int = 0
    n2w_trajectories: List = field(default_factory=list)
    w2n_trajectories: List = field(default_factory=list)
    nrem_only_trajectories: List = field(default_factory=list)
    wake_only_trajectories: List = field(default_factory=list)


@dataclass
class AblationResult:
    """Single-model ablation result with nested summary dict.

    summary structure: {trial_type: {condition: {metric: value}}}
    """
    summary: Dict = field(default_factory=dict)
    state_counts: Dict = field(default_factory=dict)
    n_trials: int = 0


# =====================================================================
#  Helpers
# =====================================================================

def get_expected_state(trial_type: str) -> str:
    return EXPECTED_STATE_MAP.get(trial_type, 'unknown')


def classify_trial_state(output: np.ndarray, target: np.ndarray,
                         n_final: int = 10) -> str:
    """Classify a single trial into wake / nrem (2-class).

    Parameters
    ----------
    output    : (T, 2) model output for response period
    target    : (T, 2) target (unused, kept for interface compatibility)
    n_final   : int    number of final time-steps to average

    Returns
    -------
    'wake' or 'nrem'
    """
    n_final = min(n_final, output.shape[0])
    final_output = output[-n_final:].mean(axis=0)

    # Channel 0 = NREM, Channel 1 = Wake → pure argmax
    output_class = np.argmax(final_output)
    return 'nrem' if output_class == 0 else 'wake'


def resolve_resp_start(meta: dict):
    """Extract temporal structure from batch metadata.

    Returns (init_steps, delay_steps, response_steps, resp_start).
    """
    init = meta['init_steps']
    delay = meta.get('delay_steps', 0)
    resp = meta['response_steps']
    return init, delay, resp, init + delay


# =====================================================================
#  State Distribution Printing
# =====================================================================

def print_state_distribution(state_counts):
    print("-" * 60)
    for trial_type in TRIAL_TYPES:
        expected = get_expected_state(trial_type)
        print(f"\n  {TRIAL_SHORT[trial_type]} (expected: {expected})")
        for condition in CONDITIONS:
            if condition not in state_counts.get(trial_type, {}):
                continue
            counts = state_counts[trial_type][condition]
            total = sum(counts.values())
            if total > 0:
                w = counts.get('wake', 0) / total * 100
                n = counts.get('nrem', 0) / total * 100
                acc = counts.get(expected, 0) / total * 100
                label = CONDITION_SHORT.get(condition, condition)
                print(f"    {label:14s}: Wake={w:5.1f}%  NREM={n:5.1f}%  "
                      f"| Acc={acc:5.1f}%")


# =====================================================================
#  Publication Style & Figure Saving
# =====================================================================

def set_publication_style():
    """Shared matplotlib rc for all analysis figures."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size':         11,
        'axes.labelsize':    12,
        'axes.titlesize':    14,
        'xtick.labelsize':   11,
        'ytick.labelsize':   11,
        'legend.fontsize':   10,
        'font.family':       'sans-serif',
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.linewidth':    0.8,
        'figure.dpi':        600,
        'savefig.dpi':       600,
        'savefig.bbox':      'tight',
        'figure.facecolor':  'white',
        'axes.facecolor':    'white',
    })


def save_figure(fig, save_path):
    """Save as PNG + mirror PDF."""
    parent = os.path.dirname(save_path) or '.'
    # os.makedirs(parent, exist_ok=True)
    # fig.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white') # we may not need the png image.
    parent = os.path.dirname(save_path) or '.'
    pdf_dir = parent + '_pdf'
    os.makedirs(pdf_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(save_path))[0]
    pdf_path = os.path.join(pdf_dir, base + '.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')


# =====================================================================
#  IO
# =====================================================================

def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"  Saved: {path}")


def load_pickle(path: str):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None