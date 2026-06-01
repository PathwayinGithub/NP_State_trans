import torch
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
from scipy.stats import mannwhitneyu, spearmanr
from tqdm import tqdm
from dataclasses import dataclass, field

from analysis_core import (
    TRIAL_TYPES, CONDITIONS, EXPECTED_STATE_MAP,
    get_expected_state, classify_trial_state, resolve_resp_start,
    TransitionData, 
)

# =====================================================================
#  FDR Correction
# =====================================================================

def _fdr_bh(pvals):
    """Benjamini-Hochberg FDR correction."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals.copy()
    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]
    ranks = np.arange(1, n + 1, dtype=float)
    adjusted = sorted_p * n / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    result = np.empty(n)
    result[sorted_idx] = adjusted
    return result


# =====================================================================
#  Sequence Metrics (Zhou et al., Neuron 2020)
# =====================================================================

def _peak_entropy(peak_times, n_timesteps, n_bins=None):
    if len(peak_times) == 0 or n_timesteps == 0:
        return 0.0
    if n_bins is None:
        n_bins = max(5, min(30, n_timesteps))
    counts, _ = np.histogram(peak_times, bins=n_bins,
                             range=(0, n_timesteps))
    p = counts / counts.sum()
    p = p[p > 0]
    H = -np.sum(p * np.log2(p))
    H_max = np.log2(n_bins)
    return float(H / H_max) if H_max > 0 else 0.0


def _temporal_sparsity(mean_act_matrix):
    n_neurons, T = mean_act_matrix.shape
    if n_neurons < 2 or T == 0:
        return 0.0
    H_max = np.log2(n_neurons)
    if H_max < 1e-12:
        return 0.0
    ts_vals = []
    for t in range(T):
        col = mean_act_matrix[:, t].copy()
        col -= col.min() 
        col = np.clip(col, 0, None) 
        s = col.sum()
        if s < 1e-12:
            continue
        p = col / s
        p = p[p > 0]
        H_t = -np.sum(p * np.log2(p))
        ts_vals.append(1.0 - H_t / H_max)
    return float(np.mean(ts_vals)) if ts_vals else 0.0


def _sequentiality_index(peak_times, mean_act_matrix, n_timesteps):
    pe = _peak_entropy(peak_times, n_timesteps)
    ts = _temporal_sparsity(mean_act_matrix)
    return float(pe * ts), pe, ts


# =====================================================================
#  Hidden Trajectory Collection (for neuron characterisation only)
# =====================================================================

def collect_hidden_trajectories(model, data_loader, device,
                                n_samples: int = 1600):
    """Collect per-trial-type hidden-state trajectories.

    These trajectories are used exclusively for neuron functional
    characterisation (Mann-Whitney U tests).  No PCA or dynamics
    analysis is performed.

    Returns
    -------
    trajectories : {trial_type: [np.ndarray of shape (time, hidden_dim)]}
    init_steps   : int
    delay_steps  : int
    """
    model.eval()
    trajectories = {trial_type: [] for trial_type in TRIAL_TYPES}
    counts = {trial_type: 0 for trial_type in TRIAL_TYPES}
    max_per_type = n_samples // 4
    init_steps = delay_steps = None

    with torch.no_grad():
        for batch in data_loader:
            trial1_inputs = batch['trial1']['inputs'].to(device,
                                                         non_blocking=True)
            trial2_inputs = batch['trial2']['inputs'].to(device,
                                                         non_blocking=True)
            metadata = batch['metadata']
            batch_size = trial1_inputs.shape[0]

            if init_steps is None:
                init_steps, delay_steps, _, _ = resolve_resp_start(
                    metadata[0])

            _, hidden1 = model(trial1_inputs, h0=None, return_hidden=True)
            h_final_trial1 = hidden1[:, -1, :]
            _, hidden2 = model(trial2_inputs, h0=h_final_trial1,
                               return_hidden=True)
            hidden2_np = hidden2.cpu().numpy()

            for b in range(batch_size):
                trial_type = metadata[b]['trial_type']
                if counts[trial_type] < max_per_type:
                    trajectories[trial_type].append(hidden2_np[b])
                    counts[trial_type] += 1

            if all(c >= max_per_type for c in counts.values()):
                break

    print(f"    Collected trajectories: {dict(counts)}")
    return trajectories, init_steps, delay_steps


# =====================================================================
#  NeuronCharacterization
# =====================================================================

@dataclass
class NeuronCharacterization:
    """Per-model neuron functional classification (4 groups).

    Groups: 0=N→W trans, 1=W→N trans, 2=NREM maint, 3=Wake maint, -1=uncl.
    """
    n_neurons: int
    seed: int

    auc_matrix: np.ndarray          # (N, 4)
    pval_corrected: np.ndarray      # (N, 4)
    dprime_matrix: np.ndarray       # (N, 4)
    group_labels: np.ndarray        # (N,)

    n2w_ids: np.ndarray
    w2n_ids: np.ndarray
    nrem_maint_ids: np.ndarray
    wake_maint_ids: np.ndarray
    unclassified_ids: np.ndarray

    sig_per_comparison: np.ndarray  # (N, 4) bool

    n2w_peak_times: np.ndarray
    w2n_peak_times: np.ndarray
    n2w_peak_entropy: float
    w2n_peak_entropy: float
    n2w_seq_index: float
    w2n_seq_index: float
    sequence_overlap_jaccard: float
    peak_order_correlation: float

    ablation_groups: Dict[str, np.ndarray]

    overlap_matrix: np.ndarray
    overlap_labels: List[str]

    correction_method: str
    temporal_windows: Dict[str, Tuple]


# =====================================================================
#  Main Characterisation
# =====================================================================

def characterize_neurons(
    trajectories: Dict[str, List[np.ndarray]],
    init_steps: int,
    delay_steps: int,
    hidden_dim: int,
    seed: int = 0,
    alpha: float = 0.01,
    correction: str = 'fdr_bh',
) -> NeuronCharacterization:
    """Classify hidden units into 4 functional groups via time-resolved MWU."""

    N = hidden_dim
    delay_end = init_steps + delay_steps

    temporal_windows = {
        'analysis': (0, delay_end),
        'init': (0, init_steps),
        'delay': (init_steps, delay_end),
    }

    n2w = trajectories.get('nrem_to_wake', [])
    w2n = trajectories.get('wake_to_nrem', [])
    nonly = trajectories.get('nrem_only', [])
    wonly = trajectories.get('wake_only', [])

    # Time windows
    bin_width = max(3, init_steps // 4)
    K_target = max(4, delay_end // bin_width)
    edges = np.unique(np.linspace(0, delay_end, K_target + 1).astype(int))
    K = len(edges) - 1
    t_centers = (edges[:-1] + edges[1:]) / 2.0

    tau_decay = max(delay_steps * 0.5, 1.0)
    trans_decay = np.exp(-np.maximum(t_centers - init_steps, 0) / tau_decay)
    maint_uniform = np.ones(K)

    print(f"    Time-resolved: K={K} windows, bin_width~{bin_width}")

    def _wmean_win(trajs, t0, t1):
        if not trajs or t0 >= t1:
            return np.zeros((0, N))
        return np.array([h[t0:t1].mean(axis=0) for h in trajs])

    def _dprime(a, b):
        mu_diff = a.mean() - b.mean()
        var_a = a.var(ddof=1) if len(a) > 1 else 0.0
        var_b = b.var(ddof=1) if len(b) > 1 else 0.0
        pooled = np.sqrt((var_a + var_b) / 2)
        return mu_diff / pooled if pooled > 1e-12 else 0.0

    comp_pairs = [
        (n2w, nonly, "N→W vs N-only"),
        (w2n, wonly, "W→N vs W-only"),
        (nonly, wonly, "N-only vs W-only"),
    ]
    n_comp = 3
    auc_3 = np.full((N, n_comp), 0.5)
    praw_3 = np.ones((N, n_comp))
    dp_3 = np.zeros((N, n_comp))

    for ci, (trajs_A, trajs_B, label) in enumerate(comp_pairs):
        nA = len(trajs_A) if trajs_A else 0
        nB = len(trajs_B) if trajs_B else 0
        if nA < 3 or nB < 3:
            print(f"    Warning: {label}: nA={nA}, nB={nB} — skipped")
            continue

        win_p = np.ones((N, K))
        win_dp = np.zeros((N, K))
        win_auc = np.full((N, K), 0.5)

        for k in range(K):
            t0, t1 = int(edges[k]), int(edges[k + 1])
            A = _wmean_win(trajs_A, t0, t1)
            B = _wmean_win(trajs_B, t0, t1)
            if A.shape[0] < 3 or B.shape[0] < 3:
                continue
            for i in range(N):
                try:
                    U, p = mannwhitneyu(A[:, i], B[:, i],
                                        alternative='two-sided')
                    win_auc[i, k] = U / (nA * nB)
                    win_p[i, k] = p
                except Exception:
                    pass
                win_dp[i, k] = _dprime(A[:, i], B[:, i])

        temporal_w = trans_decay if ci in [0, 1] else maint_uniform
        for i in range(N):
            best_k = int(np.argmin(win_p[i]))
            praw_3[i, ci] = min(float(win_p[i, best_k]) * K, 1.0)
            auc_3[i, ci] = win_auc[i, best_k]
            w = temporal_w.copy()
            w_sum = w.sum()
            if w_sum > 1e-12:
                w /= w_sum
                dp_3[i, ci] = float(np.dot(w, win_dp[i]))

    # FDR correction per comparison
    pcorr_3 = np.ones_like(praw_3)
    if correction == 'fdr_bh':
        for ci in range(n_comp):
            pcorr_3[:, ci] = _fdr_bh(praw_3[:, ci])
    elif correction == 'bonferroni':
        pcorr_3 = np.minimum(praw_3 * N, 1.0)
    else:
        pcorr_3 = praw_3.copy()
    sig_3 = pcorr_3 < alpha

    # Assignment: max |d'|
    group_labels = np.full(N, -1, dtype=int)
    for i in range(N):
        candidates = [ci for ci in range(n_comp) if sig_3[i, ci]]
        if not candidates:
            continue
        best = max(candidates, key=lambda ci: abs(dp_3[i, ci]))
        if best == 0:
            group_labels[i] = 0
        elif best == 1:
            group_labels[i] = 1
        else:
            group_labels[i] = 2 if dp_3[i, 2] > 0 else 3

    n2w_ids = np.where(group_labels == 0)[0]
    w2n_ids = np.where(group_labels == 1)[0]
    nrem_ids = np.where(group_labels == 2)[0]
    wake_ids = np.where(group_labels == 3)[0]
    uncl_ids = np.where(group_labels == -1)[0]

    # Expand to 4-column matrices
    comp_labels = ['N→W trans', 'W→N trans', 'NREM maint', 'Wake maint']
    auc_matrix = np.full((N, 4), 0.5)
    auc_matrix[:, 0] = auc_3[:, 0]
    auc_matrix[:, 1] = auc_3[:, 1]
    auc_matrix[:, 2] = auc_3[:, 2]
    auc_matrix[:, 3] = 1.0 - auc_3[:, 2]

    pval_corrected = np.ones((N, 4))
    pval_corrected[:, :3] = pcorr_3
    pval_corrected[:, 3] = pcorr_3[:, 2]

    dprime_matrix = np.zeros((N, 4))
    dprime_matrix[:, 0] = dp_3[:, 0]
    dprime_matrix[:, 1] = dp_3[:, 1]
    dprime_matrix[:, 2] = dp_3[:, 2]
    dprime_matrix[:, 3] = -dp_3[:, 2]

    sig_per_comparison = np.zeros((N, 4), dtype=bool)
    sig_per_comparison[:, 0] = sig_3[:, 0]
    sig_per_comparison[:, 1] = sig_3[:, 1]
    sig_per_comparison[:, 2] = sig_3[:, 2] & (dp_3[:, 2] > 0)
    sig_per_comparison[:, 3] = sig_3[:, 2] & (dp_3[:, 2] <= 0)

    # Sequence metrics
    def _get_delay_mat(trajs):
        if not trajs:
            return np.zeros((N, delay_end))
        return np.mean([h[:delay_end, :].T for h in trajs], axis=0)

    mean_n2w_delay = _get_delay_mat(n2w)
    mean_w2n_delay = _get_delay_mat(w2n)

    def _seq_metrics(ids, mean_mat):
        if len(ids) == 0:
            return np.array([], dtype=int), 0.0, 0.0, 0.0
        sub = mean_mat[ids]
        pt = sub.argmax(axis=1)
        sqi, pe, ts = _sequentiality_index(pt, sub, mean_mat.shape[1])
        return pt, pe, ts, sqi

    n2w_pt, n2w_pe, _, n2w_sqi = _seq_metrics(n2w_ids, mean_n2w_delay)
    w2n_pt, w2n_pe, _, w2n_sqi = _seq_metrics(w2n_ids, mean_w2n_delay)

    sig_n2w_all = np.where(sig_3[:, 0])[0]
    sig_w2n_all = np.where(sig_3[:, 1])[0]
    shared = np.intersect1d(sig_n2w_all, sig_w2n_all)
    union = np.union1d(sig_n2w_all, sig_w2n_all)
    jaccard = len(shared) / len(union) if len(union) > 0 else 0.0
    rho = 0.0
    if len(shared) >= 3:
        r, _ = spearmanr(
            mean_n2w_delay[shared].argmax(axis=1),
            mean_w2n_delay[shared].argmax(axis=1))
        rho = float(r) if not np.isnan(r) else 0.0

    # Overlap matrix
    sig_groups = [np.where(sig_per_comparison[:, ci])[0] for ci in range(4)]
    overlap = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            inter = len(np.intersect1d(sig_groups[i], sig_groups[j]))
            uni = len(np.union1d(sig_groups[i], sig_groups[j]))
            overlap[i, j] = inter / uni if uni > 0 else 0.0

    ablation_groups = {
        'n2w': n2w_ids.copy(),
        'w2n': w2n_ids.copy(),
        'nn': nrem_ids.copy(),
        'ww': wake_ids.copy(),
        'both_trans': np.union1d(n2w_ids, w2n_ids),
        'both_maint': np.union1d(nrem_ids, wake_ids),
        'all': np.where(group_labels >= 0)[0],
    }

    print(f"    Neuron counts  N→W={len(n2w_ids)}  W→N={len(w2n_ids)}  "
          f"NREM_m={len(nrem_ids)}  Wake_m={len(wake_ids)}  "
          f"Uncl={len(uncl_ids)}")

    return NeuronCharacterization(
        n_neurons=N, seed=seed,
        auc_matrix=auc_matrix, pval_corrected=pval_corrected,
        dprime_matrix=dprime_matrix, group_labels=group_labels,
        n2w_ids=n2w_ids, w2n_ids=w2n_ids,
        nrem_maint_ids=nrem_ids, wake_maint_ids=wake_ids,
        unclassified_ids=uncl_ids,
        sig_per_comparison=sig_per_comparison,
        n2w_peak_times=n2w_pt, w2n_peak_times=w2n_pt,
        n2w_peak_entropy=n2w_pe, w2n_peak_entropy=w2n_pe,
        n2w_seq_index=n2w_sqi, w2n_seq_index=w2n_sqi,
        sequence_overlap_jaccard=jaccard,
        peak_order_correlation=rho,
        ablation_groups=ablation_groups,
        overlap_matrix=overlap, overlap_labels=comp_labels,
        correction_method=correction,
        temporal_windows=temporal_windows,
    )


# =====================================================================
#  Neuron Silencing
# =====================================================================

def create_neuron_silencing(neuron_ids, hidden_dim, device='cpu'):
    """Zero out specific neurons."""
    neuron_ids = np.asarray(neuron_ids, dtype=int)
    if len(neuron_ids) == 0:
        return None, None
    mask = torch.ones(hidden_dim, device=device)
    mask[neuron_ids.tolist()] = 0.0

    def ablation_fn(h_new, h_old, direction):
        return h_new * mask.to(h_new.device).unsqueeze(0)

    dummy = torch.zeros(hidden_dim, 1, device=device)
    return ablation_fn, dummy


def create_graded_neuron_silencing(neuron_ids, hidden_dim, alpha_value,
                                   device='cpu'):
    """Graded silencing: activity * (1 - alpha_value) at target neurons."""
    neuron_ids = np.asarray(neuron_ids, dtype=int)
    if len(neuron_ids) == 0 or alpha_value == 0.0:
        return None, None
    mask = torch.ones(hidden_dim, device=device)
    mask[neuron_ids.tolist()] = 1.0 - alpha_value

    def ablation_fn(h_new, h_old, direction):
        return h_new * mask.to(h_new.device).unsqueeze(0)

    dummy = torch.zeros(hidden_dim, 1, device=device)
    return ablation_fn, dummy


def build_neuron_ablation_conditions(neuron_char, hidden_dim, device):
    """Build {condition_name: (ablation_fn, direction)} dict."""
    _map = {
        'n2w': 'ablate_n2w', 'w2n': 'ablate_w2n',
        'nn': 'ablate_nn', 'ww': 'ablate_ww',
        'both_trans': 'ablate_both',
        'all': 'ablate_all',
    }
    conditions = {'intact': (None, None)}
    for group_key, condition_name in _map.items():
        ids = neuron_char.ablation_groups.get(group_key,
                                              np.array([], dtype=int))
        conditions[condition_name] = create_neuron_silencing(
            ids, hidden_dim, device)
    return conditions


# =====================================================================
#  Temporal Window Definitions
# =====================================================================

TEMPORAL_WINDOW_LABELS = OrderedDict([
    ('full_trial',     'Full Trial'),
    ('init',           'Init Only'),
    ('delay',          'Delay Only'),
    ('init_delay',     'Init + Delay'),
    ('response',       'Response Only'),
    ('delay_response', 'Delay + Resp.'),
])


def resolve_temporal_windows(init_steps: int, delay_steps: int,
                             response_steps: int
                             ) -> 'OrderedDict[str, Tuple[int, int]]':
    """Map window names to (ablation_start_step, ablation_end_step)."""
    total = init_steps + delay_steps + response_steps
    return OrderedDict([
        ('full_trial',     (0, total)),
        ('init',           (0, init_steps)),
        ('delay',          (init_steps, init_steps + delay_steps)),
        ('init_delay',     (0, init_steps + delay_steps)),
        ('response',       (init_steps + delay_steps, total)),
        ('delay_response', (init_steps, total)),
    ])


# =====================================================================
#  TemporalAblationResult
# =====================================================================

@dataclass
class TemporalAblationResult:
    """Aggregated temporal-window × condition ablation results.

    All nested dicts: [window_name][trial_type][condition_name] → float.
    """
    window_names: List[str]
    window_ranges: Dict[str, Tuple[int, int]]
    window_labels: Dict[str, str]

    accuracy_mean:  Dict[str, Dict[str, Dict[str, float]]]
    accuracy_sem:   Dict[str, Dict[str, Dict[str, float]]]
    accuracy_std:   Dict[str, Dict[str, Dict[str, float]]]
    wake_mean:      Dict[str, Dict[str, Dict[str, float]]]
    nrem_mean:      Dict[str, Dict[str, Dict[str, float]]]

    per_model_accuracy: Dict[str, Dict[str, Dict[str, List[float]]]]

    n_models: int
    init_steps: int
    delay_steps: int
    response_steps: int

    def get_window_aggregate(self, window_name: str) -> Dict:
        """Return flat summary for one window (for printing / stats)."""
        agg: Dict = {}
        for trial_type in TRIAL_TYPES:
            agg[trial_type] = {}
            acc_map = self.accuracy_mean.get(window_name, {}).get(
                trial_type, {})
            for condition_name in acc_map:
                agg[trial_type][condition_name] = {
                    'accuracy': self.accuracy_mean.get(
                        window_name, {}).get(
                        trial_type, {}).get(condition_name, 0.0),
                    'accuracy_std': self.accuracy_std.get(
                        window_name, {}).get(
                        trial_type, {}).get(condition_name, 0.0),
                    'wake_rate': self.wake_mean.get(
                        window_name, {}).get(
                        trial_type, {}).get(condition_name, 0.0),
                    'nrem_rate': self.nrem_mean.get(
                        window_name, {}).get(
                        trial_type, {}).get(condition_name, 0.0),
                }
        agg['n_models'] = self.n_models
        return agg


# =====================================================================
#  Single-Model Temporal Ablation
# =====================================================================

def run_temporal_ablation_single(
    model,
    data_loader,
    config,
    device: str,
    neuron_char: NeuronCharacterization,
    windows: Optional[List[str]] = None,
) -> Tuple[Dict, Dict]:
    """Temporal ablation for ONE model.

    Returns
    -------
    state_counts : {window: {trial_type: {condition: {state: int}}}}
    window_ranges : OrderedDict {window: (start, end)}
    """
    model.eval()
    hidden_dim = model.hidden_dim

    conditions = build_neuron_ablation_conditions(
        neuron_char, hidden_dim, device)
    condition_names = list(conditions.keys())
    ablation_condition_names = [c for c in condition_names if c != 'intact']

    # Resolve timing
    init_steps = delay_steps = response_steps = resp_start = None
    for batch in data_loader:
        init_steps, delay_steps, response_steps, resp_start = \
            resolve_resp_start(batch['metadata'][0])
        break

    window_ranges = resolve_temporal_windows(
        init_steps, delay_steps, response_steps)
    if windows is None:
        windows = list(window_ranges.keys())

    state_counts = {
        window_name: {
            trial_type: {
                cond: {'wake': 0, 'nrem': 0}
                for cond in condition_names
            }
            for trial_type in TRIAL_TYPES
        }
        for window_name in windows
    }

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Temporal ablation', leave=False):
            trial1_inputs = batch['trial1']['inputs'].to(device)
            trial2_inputs = batch['trial2']['inputs'].to(device)
            trial2_targets = batch['trial2']['targets'].cpu().numpy()
            metadata = batch['metadata']
            batch_size = trial1_inputs.shape[0]

            # Trial 1 → get final hidden state
            _, hidden1 = model(trial1_inputs, h0=None, return_hidden=True)
            h_final_trial1 = hidden1[:, -1, :]

            # Intact forward (shared across all windows)
            intact_outputs, _ = model(trial2_inputs, h0=h_final_trial1,
                                      return_hidden=False)
            intact_outputs_np = intact_outputs.cpu().numpy()

            for b in range(batch_size):
                trial_type = metadata[b]['trial_type']
                predicted_state = classify_trial_state(
                    intact_outputs_np[b, resp_start:],
                    trial2_targets[b, resp_start:])
                for window_name in windows:
                    state_counts[window_name][trial_type][
                        'intact'][predicted_state] += 1

            # Ablation conditions × windows
            for window_name in windows:
                start, end = window_ranges[window_name]
                for condition_name in ablation_condition_names:
                    ablation_fn, ablation_dir = conditions[condition_name]
                    abl_outputs, _ = model(
                        trial2_inputs, h0=h_final_trial1,
                        return_hidden=False,
                        ablation_fn=ablation_fn,
                        ablation_direction=ablation_dir,
                        ablation_start_step=start,
                        ablation_end_step=end)
                    abl_outputs_np = abl_outputs.cpu().numpy()

                    for b in range(batch_size):
                        trial_type = metadata[b]['trial_type']
                        predicted_state = classify_trial_state(
                            abl_outputs_np[b, resp_start:],
                            trial2_targets[b, resp_start:])
                        state_counts[window_name][trial_type][
                            condition_name][predicted_state] += 1

    return state_counts, window_ranges


# =====================================================================
#  Multi-Model Temporal Ablation
# =====================================================================

def run_temporal_ablation_multi(
    models: Dict,
    data_loader,
    config,
    device: str,
    neuron_chars: Dict[int, NeuronCharacterization],
    windows: Optional[List[str]] = None,
) -> TemporalAblationResult:
    """Run temporal ablation across models and aggregate.

    Returns a single TemporalAblationResult with per-model raw values
    for downstream statistical testing.
    """
    seeds = sorted(models.keys())
    n_models = len(seeds)

    # Resolve structure from first batch
    sample_batch = next(iter(data_loader))
    init_steps, delay_steps, response_steps, _ = resolve_resp_start(
        sample_batch['metadata'][0])
    window_ranges = resolve_temporal_windows(
        init_steps, delay_steps, response_steps)
    if windows is None:
        windows = list(window_ranges.keys())

    # Condition names from first model's neuron char
    first_nc = neuron_chars[seeds[0]]
    all_condition_names = list(build_neuron_ablation_conditions(
        first_nc, first_nc.n_neurons, device).keys())

    # Per-model metric accumulators
    per_model_accuracy = {
        wn: {tt: {cn: [] for cn in all_condition_names}
             for tt in TRIAL_TYPES}
        for wn in windows
    }
    per_model_wake = {
        wn: {tt: {cn: [] for cn in all_condition_names}
             for tt in TRIAL_TYPES}
        for wn in windows
    }
    per_model_nrem = {
        wn: {tt: {cn: [] for cn in all_condition_names}
             for tt in TRIAL_TYPES}
        for wn in windows
    }

    for seed in seeds:
        model = models[seed]
        neuron_char = neuron_chars[seed]

        print(f"\n  Seed {seed}, "
              f"N_classified="
              f"{neuron_char.n_neurons - len(neuron_char.unclassified_ids)})")

        if 'cuda' in device:
            torch.cuda.empty_cache()

        state_counts, _ = run_temporal_ablation_single(
            model, data_loader, config, device, neuron_char,
            windows)

        # Accumulate per-model metrics
        for window_name in windows:
            if window_name not in state_counts:
                continue
            for trial_type in TRIAL_TYPES:
                expected = get_expected_state(trial_type)
                for condition_name in all_condition_names:
                    if condition_name not in state_counts[window_name][
                            trial_type]:
                        continue
                    counts = state_counts[window_name][trial_type][
                        condition_name]
                    total = sum(counts.values())
                    if total == 0:
                        continue
                    per_model_accuracy[window_name][trial_type][
                        condition_name].append(counts[expected] / total)
                    per_model_wake[window_name][trial_type][
                        condition_name].append(counts['wake'] / total)
                    per_model_nrem[window_name][trial_type][
                        condition_name].append(counts['nrem'] / total)

    # Aggregate: mean, SEM, std
    def _mean_sem(vals):
        if not vals:
            return 0.0, 0.0
        a = np.array(vals)
        return float(a.mean()), float(a.std(ddof=0) / np.sqrt(len(a)))

    def _std(vals):
        return float(np.std(vals, ddof=0)) if vals else 0.0

    accuracy_mean = {}
    accuracy_sem = {}
    accuracy_std = {}
    wake_mean = {}
    nrem_mean = {}

    for window_name in windows:
        accuracy_mean[window_name] = {}
        accuracy_sem[window_name] = {}
        accuracy_std[window_name] = {}
        wake_mean[window_name] = {}
        nrem_mean[window_name] = {}

        for trial_type in TRIAL_TYPES:
            accuracy_mean[window_name][trial_type] = {}
            accuracy_sem[window_name][trial_type] = {}
            accuracy_std[window_name][trial_type] = {}
            wake_mean[window_name][trial_type] = {}
            nrem_mean[window_name][trial_type] = {}

            for condition_name in all_condition_names:
                m, s = _mean_sem(per_model_accuracy[window_name][
                    trial_type][condition_name])
                accuracy_mean[window_name][trial_type][condition_name] = m
                accuracy_sem[window_name][trial_type][condition_name] = s
                accuracy_std[window_name][trial_type][condition_name] = \
                    _std(per_model_accuracy[window_name][trial_type][
                        condition_name])

                wake_mean[window_name][trial_type][condition_name], _ = \
                    _mean_sem(per_model_wake[window_name][trial_type][
                        condition_name])
                nrem_mean[window_name][trial_type][condition_name], _ = \
                    _mean_sem(per_model_nrem[window_name][trial_type][
                        condition_name])


    result = TemporalAblationResult(
        window_names=windows,
        window_ranges={wn: window_ranges[wn] for wn in windows},
        window_labels={wn: TEMPORAL_WINDOW_LABELS.get(wn, wn)
                       for wn in windows},
        accuracy_mean=accuracy_mean,
        accuracy_sem=accuracy_sem,
        accuracy_std=accuracy_std,
        wake_mean=wake_mean,
        nrem_mean=nrem_mean,
        per_model_accuracy=per_model_accuracy,
        n_models=n_models,
        init_steps=init_steps,
        delay_steps=delay_steps,
        response_steps=response_steps
    )

    return result

def collect_condition_trajectories(
    model,
    data_loader,
    device: str,
    neuron_char: NeuronCharacterization,
    n_samples_per_type: int = 400,
    ablation_start_step: int = 0,
    ablation_end_step: Optional[int] = None,
) -> Dict[str, 'TransitionData']:
    """Collect structured trajectory data for intact + all ablation conditions.

    Each trajectory dict contains:
        output_trajectory : (T, output_dim)
        trajectory        : (T, hidden_dim)
        end_state         : (hidden_dim,)
        predicted_state   : str

    Returns
    -------
    result : {condition_name: TransitionData}
        e.g. {'intact': TransitionData, 'ablate_n2w': TransitionData, ...}
    """
    model.eval()
    hidden_dim = model.hidden_dim
    conditions = build_neuron_ablation_conditions(
        neuron_char, hidden_dim, device)
    condition_names = list(conditions.keys())

    # Per-condition, per-trial-type trajectory accumulators
    traj_data = {
        cn: {tt: [] for tt in TRIAL_TYPES}
        for cn in condition_names
    }
    counts = {
        cn: {tt: 0 for tt in TRIAL_TYPES}
        for cn in condition_names
    }

    init_steps = delay_steps = response_steps = resp_start = None
    abl_end_resolved = ablation_end_step

    def _all_done():
        return all(
            all(counts[cn][tt] >= n_samples_per_type for tt in TRIAL_TYPES)
            for cn in condition_names
        )

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Collecting trajectories',
                          leave=False):
            if _all_done():
                break

            trial1_inputs = batch['trial1']['inputs'].to(device)
            trial2_inputs = batch['trial2']['inputs'].to(device)
            trial2_targets = batch['trial2']['targets'].cpu().numpy()
            metadata = batch['metadata']
            batch_size = trial1_inputs.shape[0]

            if init_steps is None:
                init_steps, delay_steps, response_steps, resp_start = \
                    resolve_resp_start(metadata[0])
                if abl_end_resolved is None:
                    abl_end_resolved = init_steps + delay_steps + response_steps

            # Trial 1 → final hidden state
            _, hidden1 = model(trial1_inputs, h0=None, return_hidden=True)
            h_final = hidden1[:, -1, :]

            for cond_name in condition_names:
                # Skip conditions that are already full
                if all(counts[cond_name][tt] >= n_samples_per_type
                       for tt in TRIAL_TYPES):
                    continue

                afn, adir = conditions[cond_name]
                outputs2, hidden2 = model(
                    trial2_inputs, h0=h_final, return_hidden=True,
                    ablation_fn=afn,
                    ablation_direction=adir,
                    ablation_start_step=ablation_start_step,
                    ablation_end_step=abl_end_resolved,
                )
                outputs2_np = outputs2.cpu().numpy()
                hidden2_np = hidden2.cpu().numpy()

                for b in range(batch_size):
                    trial_type = metadata[b]['trial_type']
                    if counts[cond_name][trial_type] >= n_samples_per_type:
                        continue
                    td = {
                        'output_trajectory': outputs2_np[b],
                        'trajectory': hidden2_np[b],
                        'end_state': hidden2_np[b, -1]
                    }
                    traj_data[cond_name][trial_type].append(td)
                    counts[cond_name][trial_type] += 1

    # Build TransitionData per condition
    result = {}
    for cond_name in condition_names:
        td_obj = TransitionData(
            init_steps=init_steps,
            delay_steps=delay_steps,
            response_steps=response_steps,
        )
        td_obj.n2w_trajectories = traj_data[cond_name]['nrem_to_wake']
        td_obj.w2n_trajectories = traj_data[cond_name]['wake_to_nrem']
        td_obj.nrem_only_trajectories = traj_data[cond_name]['nrem_only']
        td_obj.wake_only_trajectories = traj_data[cond_name]['wake_only']
        result[cond_name] = td_obj

    total_collected = sum(
        sum(counts[cn][tt] for tt in TRIAL_TYPES)
        for cn in condition_names
    )
    print(f"    Collected {total_collected} trajectory samples "
          f"across {len(condition_names)} conditions")
    return result