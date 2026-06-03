import os
import pickle
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from tqdm import tqdm

from model import ContinuousTimeRNN
from config import ExperimentConfig
from analysis_core import (
    TRIAL_TYPES, TRIAL_COLORS, TRIAL_SHORT,
    EXPECTED_STATE_MAP,
    resolve_resp_start,
    classify_trial_state, set_publication_style,
)
from analysis_core import save_figure
from analysis_neurons import (
    NeuronCharacterization,
    build_neuron_ablation_conditions,
)
from matplotlib.patches import Rectangle


# ═══════════════════════════════════════════════════════════════════
#  Error-trial display constant 
# ═══════════════════════════════════════════════════════════════════

ERROR_DISPLAY: str = 'error'


# ═══════════════════════════════════════════════════════════════════
#  Module-local constants
# ═══════════════════════════════════════════════════════════════════

FONT = {
    'suptitle': 16, 'title': 14, 'label': 12,
    'tick': 11, 'legend': 10, 'annot': 10, 'bar_text': 10,
}

ABLATION_ROWS = OrderedDict([
    ('ablate_w2n',  'Cut W→N'),
    ('ablate_n2w',  'Cut N→W'),
    ('ablate_both', 'Cut Both'),
    ('ablate_nn', 'Cut N→N'),
    ('ablate_ww', 'Cut W→W'),
])

ROW_HIGHLIGHT = {
    'ablate_w2n':  ['wake_to_nrem'],
    'ablate_n2w':  ['nrem_to_wake'],
    'ablate_both': ['nrem_to_wake', 'wake_to_nrem'],
    'ablate_nn': ['nrem_only'],
    'ablate_ww': ['wake_only'],
}

# ═══════════════════════════════════════════════════════════════════
#  Input-conditioned flow-field constants  (4-channel one-hot)
# ═══════════════════════════════════════════════════════════════════

TRIAL_TYPE_INIT_INPUT = {
    'nrem_to_wake': [1.0, 1.0, 0.0],
    'wake_to_nrem': [0.0, 0.0, 1.0],
    'nrem_only':    [0.0, 1.0, 0.0],
    'wake_only':    [1.0, 0.0, 1.0],
}

INPUT_COL_ORDER = ['x0', 'nrem_to_wake', 'wake_to_nrem', 'nrem_only', 'wake_only']

INPUT_COL_LABELS = {
    'x0':             'x = 0  (Autonomous)',
    'nrem_to_wake':   'x = N→W Init',
    'wake_to_nrem':   'x = W→N Init',
    'nrem_only':      'x = N-Only Init',
    'wake_only':      'x = W-Only Init',
}

ROW_ACTIVE_INPUT_COLS = {
    'ablate_w2n':  ['x0', 'wake_to_nrem'],
    'ablate_n2w':  ['x0', 'nrem_to_wake'],
    'ablate_both': ['x0', 'nrem_to_wake', 'wake_to_nrem'],
    'ablate_nn':   ['x0', 'nrem_only'],
    'ablate_ww':   ['x0', 'wake_only'],
}


DYNAMICS_WINDOWS = OrderedDict([
    ('init_delay', 'Init + Delay'),
    ('full_trial', 'Full Trial'),
])

GRID_RES_DEFAULT = 60
N_TRAJ_PER_TYPE  = 400
CMAP_NAME        = 'RdYlBu_r'

CMAP_LANDSCAPE = mcolors.LinearSegmentedColormap.from_list(
    'flow_q', ['#5B9FD4', '#F4EFE9', '#D95F4B'], N=512)

SLOW_MERGE_RADIUS = 0.3
SLOW_ATTR_RADIUS  = 1.0

PCA_FIT_TRIAL_TYPES = ('nrem_to_wake', 'wake_to_nrem')#, 'nrem_only', 'wake_only') #  we modified here for considering only transition trials. 

_PHASE_PROPS = OrderedDict([
    ('init',     {'ls': ':',  'lw_f': 0.50, 'a_f': 0.50, 'arrow_scale': 5}),
    ('delay',    {'ls': '--', 'lw_f': 0.72, 'a_f': 0.72, 'arrow_scale': 6}),
    ('response', {'ls': '-',  'lw_f': 1.00, 'a_f': 1.00, 'arrow_scale': 8}),
])

N_INDIVIDUAL_SHOW = 20
N_QUIVER_DEFAULT  = 35

EXTEND_N_STEPS    = 50
EXTEND_COLOR      = '#00E676'
EXTEND_LW_IND     = 0.55
EXTEND_LW_MEAN    = 2.8
EXTEND_ALPHA_IND  = 0.18
EXTEND_ALPHA_MEAN = 0.85

QUIVER_MODE    = 'traj_const_z'
QUIVER_N_STEPS = 5
# ═══════════════════════════════════════════════════════════════════
#  Phase colour helper
# ═══════════════════════════════════════════════════════════════════

def _phase_adjusted_color(base_color: str, phase: str):
    c = np.array(mcolors.to_rgb(base_color))
    if phase == 'init':
        return tuple(np.clip(c * 0.58 + 0.42, 0.0, 1.0))
    elif phase == 'delay':
        return tuple(np.clip(c * 0.80 + 0.20, 0.0, 1.0))
    return tuple(c)


# ═══════════════════════════════════════════════════════════════════
#  Cache / IO helpers
# ═══════════════════════════════════════════════════════════════════

def _cache_dir(config: ExperimentConfig) -> str:
    d = os.path.join(config.paths.data_dir, 'analysis', 'dynamics')
    os.makedirs(d, exist_ok=True)
    return d


def _fig_dir(config: ExperimentConfig) -> str:
    d = os.path.join(config.paths.figure_dir, 'dynamics')
    os.makedirs(d, exist_ok=True)
    return d


def _save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"    cached → {os.path.basename(path)}")


def _load(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


# ═══════════════════════════════════════════════════════════════════
#  Trial filtering: keep only ERROR trials (predicted ≠ expected)
# ═══════════════════════════════════════════════════════════════════

def filter_error_trials(
    trajectory_list: List[np.ndarray],
    predicted_states: List[str],
    trial_type: str,
) -> Tuple[List[np.ndarray], int, int]:
    """Keep trajectories whose prediction disagrees with the expected state."""
    expected = EXPECTED_STATE_MAP[trial_type]
    assert len(trajectory_list) == len(predicted_states)
    filtered = [arr for arr, p in zip(trajectory_list, predicted_states)
                if p != expected]
    return filtered, len(filtered), len(trajectory_list)


def _count_errors(preds: List[str], trial_type: str) -> int:
    expected = EXPECTED_STATE_MAP[trial_type]
    return sum(1 for p in preds if p != expected)


def _check_error_coverage(
    pred_states: Dict[str, Dict[str, List[str]]],
    min_errors: int = 1,
) -> Tuple[bool, Dict[str, Dict[str, int]]]:
    detail: Dict[str, Dict[str, int]] = {}
    all_ok = True
    for cond, hl_tts in ROW_HIGHLIGHT.items():
        detail[cond] = {}
        preds_cond = pred_states.get(cond, {})
        for tt in hl_tts:
            preds = preds_cond.get(tt, [])
            n_err = _count_errors(preds, tt)
            detail[cond][tt] = n_err
            if n_err < min_errors:
                all_ok = False
    return all_ok, detail


def _plottable_conditions(pred_states, min_errors=1):
    plottable, skipped = [], []
    for cond, hl_tts in ROW_HIGHLIGHT.items():
        preds_cond = pred_states.get(cond, {})
        if cond == 'ablate_both':
            ok = False
            for tt in hl_tts:
                preds = preds_cond.get(tt, [])
                if _count_errors(preds, tt) >= min_errors:
                    ok = True
                    break
        else:
            ok = True
            for tt in hl_tts:
                preds = preds_cond.get(tt, [])
                if _count_errors(preds, tt) < min_errors:
                    ok = False
                    break
        (plottable if ok else skipped).append(cond)
    return plottable, skipped


def _print_error_coverage(
    seed: int,
    detail: Dict[str, Dict[str, int]],
    all_ok: bool,
    min_errors: int = 1,
):
    status = "✓ ALL PASS" if all_ok else "△ PARTIAL"
    print(f"\n    Error coverage (seed={seed}, "
          f"min={min_errors}): {status}")
    print(f"    {'Condition':<16} {'Trial Type':<12} "
          f"{'#Error':>10} {'Status':>8}")
    print(f"    {'─' * 50}")
    for cond, tt_counts in detail.items():
        label = ABLATION_ROWS.get(cond, cond)
        for tt, n_err in tt_counts.items():
            tt_lab = TRIAL_SHORT.get(tt, tt)
            ok = "✓ plot" if n_err >= min_errors else "– skip"
            print(f"    {label:<16} {tt_lab:<12} {n_err:>10} {ok:>8}")


def summarise_error_counts(
    pred_states: Dict[str, Dict[str, List[str]]],
    condition: str,
) -> Tuple[int, int]:
    """Return (n_errors, n_total) across all trial types for one condition."""
    n_errors = 0
    n_total = 0
    for tt in TRIAL_TYPES:
        expected = EXPECTED_STATE_MAP[tt]
        for ps in pred_states.get(condition, {}).get(tt, []):
            n_total += 1
            if ps != expected:
                n_errors += 1
    return n_errors, n_total


# ═══════════════════════════════════════════════════════════════════
#  Robust neuron-ID extraction
# ═══════════════════════════════════════════════════════════════════

_NEURON_ID_ATTR_CANDIDATES = {
    'n2w': ['n2w_ids', 'n2w_neuron_ids', 'transition_n2w_ids'],
    'w2n': ['w2n_ids', 'w2n_neuron_ids', 'transition_w2n_ids'],
    'ww':  ['ww_ids', 'wake_maintenance_ids', 'w_maint_ids'],
    'nn':  ['nn_ids', 'nrem_maintenance_ids', 'n_maint_ids'],
}


def _resolve_neuron_ids(
    neuron_char: NeuronCharacterization,
    group_key: str,
) -> np.ndarray:
    if group_key == 'both_trans':
        n2w = _resolve_neuron_ids(neuron_char, 'n2w')
        w2n = _resolve_neuron_ids(neuron_char, 'w2n')
        combined = np.concatenate([n2w, w2n])
        return np.unique(combined).astype(int)
    abl_groups = getattr(neuron_char, 'ablation_groups', None)
    if isinstance(abl_groups, dict) and group_key in abl_groups:
        ids = abl_groups[group_key]
        if ids is not None and hasattr(ids, '__len__'):
            arr = np.asarray(ids, dtype=int).ravel()
            if arr.size > 0:
                return arr
    for attr_name in _NEURON_ID_ATTR_CANDIDATES.get(group_key, []):
        ids = getattr(neuron_char, attr_name, None)
        if ids is not None and hasattr(ids, '__len__'):
            arr = np.asarray(ids, dtype=int).ravel()
            if arr.size > 0:
                return arr
    return np.array([], dtype=int)


# ═══════════════════════════════════════════════════════════════════
#  1. Trajectory collection
# ═══════════════════════════════════════════════════════════════════

def _resolve_abl_window(window_name: str, init_s: int,
                         delay_s: int, total_s: int) -> Tuple[int, int]:
    if window_name == 'init_delay':
        return 0, init_s + delay_s
    return 0, total_s


def collect_dynamics_trajectories(
    model: ContinuousTimeRNN,
    data_loader,
    neuron_char: NeuronCharacterization,
    device: str,
    ablation_window: str = 'init_delay',
    n_per_type: int = N_TRAJ_PER_TYPE,
) -> Tuple[Dict, Dict, Dict]:
    """Collect per-condition, per-trial-type hidden trajectories."""
    model.eval()
    H = model.hidden_dim
    conds = build_neuron_ablation_conditions(neuron_char, H, device)
    cond_names = list(conds.keys())

    resp_start = None
    for batch in data_loader:
        init_s, delay_s, resp_s, resp_start = resolve_resp_start(
            batch['metadata'][0])
        break
    total_s = init_s + delay_s + resp_s

    timing = dict(
        init_steps=init_s, delay_steps=delay_s,
        response_steps=resp_s, total_steps=total_s,
        resp_start=resp_start,
    )
    abl_s, abl_e = _resolve_abl_window(ablation_window, init_s,
                                         delay_s, total_s)

    trajs       = {cn: {tt: [] for tt in TRIAL_TYPES} for cn in cond_names}
    pred_states = {cn: {tt: [] for tt in TRIAL_TYPES} for cn in cond_names}
    cnts        = {cn: {tt: 0  for tt in TRIAL_TYPES} for cn in cond_names}

    def _all_done():
        return all(
            all(cnts[cn][tt] >= n_per_type for tt in TRIAL_TYPES)
            for cn in cond_names)

    with torch.no_grad():
        for batch in tqdm(data_loader,
                          desc=f'  trajs ({ablation_window})', leave=False):
            t1 = batch['trial1']['inputs'].to(device)
            t2 = batch['trial2']['inputs'].to(device)
            t2_tgt = batch['trial2']['targets'].cpu().numpy()
            meta = batch['metadata']
            bs = t1.shape[0]

            _, h1 = model(t1, h0=None, return_hidden=True)
            hf1 = h1[:, -1, :]
            hf1_np = hf1.cpu().numpy()

            for cn in cond_names:
                if all(cnts[cn][tt] >= n_per_type for tt in TRIAL_TYPES):
                    continue
                afn, adir = conds[cn]
                kw = {}
                if cn != 'intact':
                    kw = dict(ablation_fn=afn, ablation_direction=adir,
                              ablation_start_step=abl_s,
                              ablation_end_step=abl_e)
                out2, h2 = model(t2, h0=hf1, return_hidden=True, **kw)
                h2_np  = h2.cpu().numpy()
                out2_np = out2.cpu().numpy()

                for b in range(bs):
                    tt = meta[b]['trial_type']
                    if cnts[cn][tt] < n_per_type:
                        arr = np.concatenate(
                            [hf1_np[b:b + 1], h2_np[b]], axis=0)
                        trajs[cn][tt].append(arr)
                        o_r = out2_np[b, resp_start:]
                        t_r = t2_tgt[b, resp_start:]
                        ps = classify_trial_state(o_r, t_r)
                        pred_states[cn][tt].append(ps)
                        cnts[cn][tt] += 1

            if _all_done():
                break

    n_all = sum(sum(cnts[c][t] for t in TRIAL_TYPES) for c in cond_names)
    print(f"    collected: {n_all} total  "
          f"(resp_start={resp_start}, total_steps={total_s})")
    for cn in cond_names:
        if cn == 'intact':
            continue
        n_err, n_tot = summarise_error_counts(pred_states, cn)
        if n_tot > 0:
            print(f"      {cn}: {n_err}/{n_tot} {ERROR_DISPLAY} "
                  f"({100 * n_err / n_tot:.1f}%)")

    return trajs, timing, pred_states


# ═══════════════════════════════════════════════════════════════════
#  2. PCA
# ═══════════════════════════════════════════════════════════════════

def fit_dynamics_pca(
    trajs: Dict, timing: Dict, n_components: int = 3,
) -> PCA:
    pts = []
    intact_data = trajs.get('intact', {})
    if not intact_data:
        raise ValueError("No 'intact' condition found in trajs")
    skip = timing.get('init_steps', 0) + 1
    n_used = 0
    for tt in PCA_FIT_TRIAL_TYPES:
        arr_list = intact_data.get(tt, [])
        for arr in arr_list:
            pts.append(arr[skip:])
        n_used += len(arr_list)
    if not pts:
        raise ValueError(
            f"No intact trajectories for {PCA_FIT_TRIAL_TYPES}")
    X = np.vstack(pts)
    n_comp = min(n_components, *X.shape)
    pca = PCA(n_components=n_comp).fit(X)
    ev = pca.explained_variance_ratio_
    print(f"    PCA (intact, skip={skip}): "
          f"{n_used} trajs, {X.shape[0]:,} pts × {X.shape[1]}D")
    print(f"      ▸ PC1 = {ev[0]:.2%}")
    print(f"      ▸ PC2 = {ev[1]:.2%}")
    print(f"      ▸ PC1+PC2 = {ev[:2].sum():.2%}")
    if len(ev) >= 3:
        print(f"      ▸ PC1+PC2+PC3 = {ev[:3].sum():.2%}")
    if ev[:2].sum() < 0.50:
        print(f"      ⚠  PC1+PC2 < 50% — strongly recommend using PLS dimensionality reduction")
    elif ev[:2].sum() < 0.70:
        print(f"      △  PC1+PC2 < 70% — recommend validating with PLS")
    else:
        print(f"      ✓  PC1+PC2 ≥ 70% — 2D is sufficient")
    return pca


# ═══════════════════════════════════════════════════════════════════
#  3. Grid-based flow field
# ═══════════════════════════════════════════════════════════════════

def _get_pc_range(trajs: Dict, pca: PCA, timing: Dict,
                  ) -> Tuple[float, float, float, float]:
    all_pc = []
    for cn_data in trajs.values():
        for arr_list in cn_data.values():
            for arr in arr_list:
                pc = pca.transform(arr)[:, :2]
                all_pc.append(pc)
    all_pc = np.vstack(all_pc)
    return (all_pc[:, 0].min(), all_pc[:, 0].max(),
            all_pc[:, 1].min(), all_pc[:, 1].max())


def _build_trajectory_cloud(
    trajs: Dict, pca: PCA
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (pc_cloud [N, n_comp], h_cloud [N, hidden_dim])."""
    pts = []
    intact_data = trajs.get('intact', {})
    for tt in TRIAL_TYPES:
        for arr in intact_data.get(tt, []):
            pts.append(arr)
    if not pts:
        n_comp = pca.n_components_
        return (np.empty((0, n_comp), dtype=np.float32),
                np.empty((0, 0), dtype=np.float32))
    all_h  = np.vstack(pts).astype(np.float32)
    all_pc = pca.transform(all_h).astype(np.float32)
    return all_pc, all_h


def _reconstruct_grid_adaptive(
    pc1_vals: np.ndarray,
    pc2_vals: np.ndarray,
    pca: PCA,
    traj_cloud_pc: Optional[np.ndarray] = None,
    k_neighbors: int = 50,
    sigma_weight: float = 1.5,
) -> np.ndarray:
    grid_res_x = len(pc1_vals)
    grid_res_y = len(pc2_vals)
    Xi, Yi = np.meshgrid(pc1_vals, pc2_vals)
    n_grid = grid_res_x * grid_res_y
    n_comp = pca.n_components_

    grid_pc = np.zeros((n_grid, n_comp), dtype=np.float32)
    grid_pc[:, 0] = Xi.ravel()
    grid_pc[:, 1] = Yi.ravel()

    if (traj_cloud_pc is not None
            and len(traj_cloud_pc) >= k_neighbors
            and n_comp > 2):
        tree = cKDTree(traj_cloud_pc[:, :2])
        grid_xy = grid_pc[:, :2]
        dists, idxs = tree.query(grid_xy, k=k_neighbors)
        weights = np.exp(-0.5 * (dists / sigma_weight) ** 2)
        weights_sum = weights.sum(axis=1, keepdims=True)
        weights_sum = np.where(weights_sum > 1e-12, weights_sum, 1.0)
        weights_norm = weights / weights_sum
        for pc_i in range(2, n_comp):
            neighbor_vals = traj_cloud_pc[idxs, pc_i]
            grid_pc[:, pc_i] = (weights_norm * neighbor_vals).sum(axis=1)
        print(f"      grid PC3+ reconstructed from {len(traj_cloud_pc)} "
              f"trajectory points (k={k_neighbors}, σ={sigma_weight:.1f})")
    else:
        if n_comp > 2:
            print(f"      grid PC3+ = 0 (no trajectory cloud available)")

    h_grid = pca.inverse_transform(grid_pc).astype(np.float32)
    return h_grid

def _reconstruct_grid_linear(
    pc1_vals: np.ndarray,
    pc2_vals: np.ndarray,
    pca: PCA,
) -> np.ndarray:
    Xi, Yi = np.meshgrid(pc1_vals, pc2_vals)
    grid_pc = np.zeros((Xi.size, pca.n_components_), dtype=np.float32)
    grid_pc[:, 0] = Xi.ravel()
    grid_pc[:, 1] = Yi.ravel()
    return pca.inverse_transform(grid_pc).astype(np.float32)


def _autonomous_step_maybe_ablated(
    model: ContinuousTimeRNN,
    h: torch.Tensor,
    ablation_fn=None,
    ablation_dir=None,
    x_const: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if x_const is not None:
        x = x_const.expand(h.shape[0], -1)
        _, h_next = model.single_step(
            x, h,
            ablation_fn=ablation_fn,
            ablation_direction=ablation_dir)
        return h_next
    if ablation_fn is None and ablation_dir is None:
        return model.autonomous_step(h)
    x = torch.zeros(h.shape[0], model.input_dim, device=h.device)
    _, h_next = model.single_step(
        x, h,
        ablation_fn=ablation_fn,
        ablation_direction=ablation_dir)
    return h_next


def _compute_q_batch(
    model: ContinuousTimeRNN,
    h_points: np.ndarray,
    device: str,
    batch_size: int = 4096,
    ablation_fn=None,
    ablation_dir=None,
    x_const: Optional[torch.Tensor] = None,
) -> np.ndarray:
    model.eval()
    parts: List[np.ndarray] = []
    for i0 in range(0, len(h_points), batch_size):
        i1 = min(i0 + batch_size, len(h_points))
        h = torch.tensor(
            h_points[i0:i1], dtype=torch.float32, device=device)
        with torch.no_grad():
            h_next = _autonomous_step_maybe_ablated(
                model, h, ablation_fn, ablation_dir, x_const=x_const)
            dh = h_next - h
            q = 0.5 * (dh ** 2).sum(dim=-1)
            parts.append(q.cpu().numpy())
    return np.concatenate(parts)


def _compute_pc_displacement_batch(
    model: ContinuousTimeRNN,
    h_points: np.ndarray,
    pca: PCA,
    device: str,
    n_steps: int = 1,
    batch_size: int = 4096,
    ablation_fn=None,
    ablation_dir=None,
    x_const: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    V2 = pca.components_[:2]
    du_parts: List[np.ndarray] = []
    dv_parts: List[np.ndarray] = []
    for i0 in range(0, len(h_points), batch_size):
        i1 = min(i0 + batch_size, len(h_points))
        h0 = torch.tensor(
            h_points[i0:i1], dtype=torch.float32, device=device)
        with torch.no_grad():
            h = h0.clone()
            for _ in range(n_steps):
                h = _autonomous_step_maybe_ablated(
                    model, h, ablation_fn, ablation_dir, x_const=x_const)
            dh = (h - h0).cpu().numpy()
        dh_pc = dh @ V2.T
        du_parts.append(dh_pc[:, 0])
        dv_parts.append(dh_pc[:, 1])
    return np.concatenate(du_parts), np.concatenate(dv_parts)


def _compute_pc_velocity_batch(
    model, h_points, pca, device, batch_size=4096,
    ablation_fn=None, ablation_dir=None, x_const=None,
):
    return _compute_pc_displacement_batch(
        model, h_points, pca, device,
        n_steps=1, batch_size=batch_size,
        ablation_fn=ablation_fn, ablation_dir=ablation_dir,
        x_const=x_const,
    )

def _reconstruct_grid_traj_const_z(
    pc1_vals: np.ndarray,
    pc2_vals: np.ndarray,
    pca: PCA,
    traj_cloud_pc: Optional[np.ndarray] = None,
) -> np.ndarray:
    
    Xi, Yi = np.meshgrid(pc1_vals, pc2_vals)
    n_grid = Xi.size
    n_comp = pca.n_components_

    grid_pc = np.zeros((n_grid, n_comp), dtype=np.float32)
    grid_pc[:, 0] = Xi.ravel()
    grid_pc[:, 1] = Yi.ravel()

    if (traj_cloud_pc is not None
            and traj_cloud_pc.shape[0] > 0
            and traj_cloud_pc.shape[1] >= n_comp
            and n_comp > 2):
        z_const = np.median(traj_cloud_pc[:, 2:], axis=0)
        for k in range(2, n_comp):
            grid_pc[:, k] = z_const[k - 2]
        print(f"      const-z slice: PC3+ median = "
              f"{np.array2string(z_const, precision=2)}")

    h_grid = pca.inverse_transform(grid_pc).astype(np.float32)
    return h_grid

# ═══════════════════════════════════════════════════════════════════
#  Speed-proportional quiver helper 
# ═══════════════════════════════════════════════════════════════════

def _draw_speed_quiver(ax, Xq, Yq, U_raw, V_raw):
    """
    Draw a quiver plot where arrow length ∝ speed (log-compressed).

    Mapping
    -------
    • 85th-percentile speed  →  arrow spans ~1 grid cell (reference length)
    • Slower regions         →  proportionally shorter arrows
    • Faster regions         →  proportionally longer (capped at 2.5×)
    • Near-zero speed        →  arrow hidden (mag < 0.04 threshold)

    The log1p compression keeps slow-region arrows visible while still
    making the fast/slow contrast clearly legible.
    """
    finite = np.isfinite(U_raw) & np.isfinite(V_raw)
    spd = np.sqrt(
        np.where(finite, U_raw, 0.0) ** 2 +
        np.where(finite, V_raw, 0.0) ** 2
    )

    valid_spd = spd[finite & (spd > 1e-10)]
    ref = float(np.percentile(valid_spd, 85)) if valid_spd.size > 0 else 1.0

    # ── log1p magnitude: 0→0, ref→1, monotonically growing ──
    # log1p(spd/ref) / log1p(1)  gives exactly 1.0 when spd == ref
    log_norm = np.log1p(1.0)                           # ≈ 0.6931
    mag = np.log1p(spd / (ref + 1e-12)) / log_norm     # 0 at spd=0, 1 at ref

    # hard-clip upper end so outliers don't dominate the grid
    mag = np.clip(mag, 0.0, 2.5)

    # hide near-zero arrows (attractors / fixed points)
    mag = np.where(mag < 0.02, 0.0, mag)

    # unit direction × log-magnitude
    spd_safe = np.where(spd > 1e-12, spd, 1.0)
    Ux = (U_raw / spd_safe) * mag
    Vy = (V_raw / spd_safe) * mag
    Ux = np.where(finite, Ux, np.nan)
    Vy = np.where(finite, Vy, np.nan)

    # scale so that mag=1 arrow spans ~1 grid cell
    pc1_min, pc1_max = Xq.min(), Xq.max()
    pc2_min, pc2_max = Yq.min(), Yq.max()
    n_cols = Xq.shape[1] - 1
    n_rows = Yq.shape[0] - 1
    grid_step = max(
        (pc1_max - pc1_min) / max(n_cols, 1),
        (pc2_max - pc2_min) / max(n_rows, 1),
    )
    # mag=1 arrow = 1 grid cell; capped arrows (2.5×) = 2.5 cells
    scale_xy = 1.0 / (grid_step + 1e-12)

    ax.quiver(
        Xq, Yq, Ux, Vy,
        color='#3b3b3b', alpha=0.80,
        width=0.0028,
        headwidth=3.8,
        headlength=4.5,
        headaxislength=4.0,
        minshaft=1.2,
        minlength=0.0,       # allow zero-length (hidden) arrows
        scale=scale_xy,
        scale_units='xy',
        angles='xy',
        pivot='tail',
        zorder=2,
    )


def compute_flow_field(
    model: ContinuousTimeRNN,
    pca: PCA,
    device: str,
    pc_range: Tuple[float, float, float, float],
    grid_res: int = GRID_RES_DEFAULT,
    padding: float = 0.15,
    ablation_fn=None,
    ablation_dir=None,
    batch_size: int = 4096,
    smooth_sigma: float = 1.0,
    n_quiver: int = N_QUIVER_DEFAULT,
    compute_quiver: bool = True,
    x_const: Optional[torch.Tensor] = None,
    traj_cloud_pc: Optional[np.ndarray] = None,
    traj_cloud_h:  Optional[np.ndarray] = None,
) -> Dict:
    model.eval()
    pc1_min, pc1_max, pc2_min, pc2_max = pc_range
    pad1 = (pc1_max - pc1_min) * padding
    pad2 = (pc2_max - pc2_min) * padding

    pc1_v = np.linspace(pc1_min - pad1, pc1_max + pad1, grid_res)
    pc2_v = np.linspace(pc2_min - pad2, pc2_max + pad2, grid_res)
    Xi, Yi = np.meshgrid(pc1_v, pc2_v)

    h_approx = _reconstruct_grid_adaptive(
        pc1_v, pc2_v, pca, traj_cloud_pc)
    
    q_values = _compute_q_batch(
        model, h_approx, device, batch_size,
        ablation_fn, ablation_dir, x_const=x_const)
    q_grid = q_values.reshape(grid_res, grid_res)

    speed_grid = np.sqrt(2.0 * q_grid)
    log_speed = np.log10(speed_grid + 1e-12)
    Sg = gaussian_filter(log_speed, sigma=smooth_sigma)
    Sc = Sg

    result = dict(
        Xi=Xi, Yi=Yi, Sc=Sc, log_speed_raw=log_speed,
        pc1_vals=pc1_v, pc2_vals=pc2_v,
        pc1_grid=Xi, pc2_grid=Yi,
        speed=speed_grid, speed_full=speed_grid,
    )

    if compute_quiver:
        Xq, Yq, U_dir, V_dir = _compute_quiver_field(
            model, pca, device, pc_range, n_quiver, padding,
            batch_size, ablation_fn, ablation_dir,
            x_const=x_const,
            traj_cloud_pc=traj_cloud_pc,
            traj_cloud_h=traj_cloud_h)
        result.update(Xq=Xq, Yq=Yq, U_dir=U_dir, V_dir=V_dir)

    return result

def _compute_quiver_field(
    model: ContinuousTimeRNN,
    pca: PCA,
    device: str,
    pc_range: Tuple[float, float, float, float],
    n_quiver: int = N_QUIVER_DEFAULT,
    padding: float = 0.15,
    batch_size: int = 4096,
    ablation_fn=None,
    ablation_dir=None,
    x_const: Optional[torch.Tensor] = None,
    traj_cloud_pc: Optional[np.ndarray] = None,
    traj_cloud_h:  Optional[np.ndarray] = None,
    mode: Optional[str] = None,
    n_steps: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if mode is None:
        mode = QUIVER_MODE
    if n_steps is None:
        n_steps = QUIVER_N_STEPS

    pc1_min, pc1_max, pc2_min, pc2_max = pc_range
    pad1 = (pc1_max - pc1_min) * padding
    pad2 = (pc2_max - pc2_min) * padding
    xi = np.linspace(pc1_min - pad1, pc1_max + pad1, n_quiver)
    yi = np.linspace(pc2_min - pad2, pc2_max + pad2, n_quiver)
    Xq, Yq = np.meshgrid(xi, yi)

    if mode == 'traj_const_z':
        h_grid = _reconstruct_grid_traj_const_z(
            xi, yi, pca, traj_cloud_pc)
    elif mode == 'adaptive':
        h_grid = _reconstruct_grid_adaptive(
            xi, yi, pca, traj_cloud_pc,
            k_neighbors=24, sigma_weight=1.0)
    elif mode == 'linear':
        h_grid = _reconstruct_grid_linear(xi, yi, pca)
    else:
        raise ValueError(
            f"Unknown QUIVER_MODE='{mode}'; "
            f"choose from 'traj_const_z' | 'adaptive' | 'linear'")

    dU, dV = _compute_pc_displacement_batch(
        model, h_grid, pca, device,
        n_steps=n_steps,
        batch_size=batch_size,
        ablation_fn=ablation_fn,
        ablation_dir=ablation_dir,
        x_const=x_const,
    )
    U = dU.reshape(n_quiver, n_quiver)
    V = dV.reshape(n_quiver, n_quiver)
    return Xq, Yq, U, V


def compute_input_conditioned_flows(
    model: ContinuousTimeRNN,
    pca: PCA,
    device: str,
    pc_range: Tuple,
    grid_res: int = GRID_RES_DEFAULT,
    traj_cloud_pc: Optional[np.ndarray] = None,
    traj_cloud_h:  Optional[np.ndarray] = None,
) -> Dict[str, Dict]:
    flows: Dict[str, Dict] = {}
    items = list(TRIAL_TYPE_INIT_INPUT.items())
    for i, (tt, x_vec) in enumerate(items, 1):
        print(f"    [{i}/{len(items)}] input-conditioned flow: "
              f"x = {tt}  {x_vec}")
        x_t = torch.tensor([x_vec], dtype=torch.float32, device=device)
        flows[tt] = compute_flow_field(
            model, pca, device, pc_range, grid_res,
            x_const=x_t, compute_quiver=True,
            traj_cloud_pc=traj_cloud_pc,
            traj_cloud_h=traj_cloud_h)
    return flows


# ═══════════════════════════════════════════════════════════════════
#  5. Attractor centres
# ═══════════════════════════════════════════════════════════════════

def _compute_attractor_centers(trajs_intact: Dict, pca: PCA) -> Dict:
    nrem_ep, wake_ep = [], []
    for arr in trajs_intact.get('nrem_only', []):
        nrem_ep.append(arr[-1])
    for arr in trajs_intact.get('wake_only', []):
        wake_ep.append(arr[-1])
    for arr in trajs_intact.get('nrem_only', []):
        nrem_ep.append(arr[0])
    for arr in trajs_intact.get('wake_only', []):
        wake_ep.append(arr[0])
    for arr in trajs_intact.get('wake_to_nrem', []):
        nrem_ep.append(arr[-1])
    for arr in trajs_intact.get('nrem_to_wake', []):
        wake_ep.append(arr[-1])
    for arr in trajs_intact.get('wake_to_nrem', []):
        wake_ep.append(arr[0])
    for arr in trajs_intact.get('nrem_to_wake', []):
        nrem_ep.append(arr[0])

    centers = {}
    if nrem_ep:
        pcs = pca.transform(np.array(nrem_ep))[:, :2]
        pc_mean = pcs.mean(axis=0)
        centers['NREM'] = (float(pc_mean[0]), float(pc_mean[1]))
    if wake_ep:
        pcs = pca.transform(np.array(wake_ep))[:, :2]
        pc_mean = pcs.mean(axis=0)
        centers['Wake'] = (float(pc_mean[0]), float(pc_mean[1]))

    return centers


# ═══════════════════════════════════════════════════════════════════
#  6. Trajectory → PC helpers
# ═══════════════════════════════════════════════════════════════════

def _mean_traj_pc(
    trajs_list: List[np.ndarray],
    pca: PCA,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> Optional[np.ndarray]:
    if not trajs_list:
        return None
    pcs = []
    for arr in trajs_list:
        seg = arr[start_idx:end_idx]
        if len(seg) < 2:
            continue
        pc = pca.transform(seg)[:, :2]
        pcs.append(pc)
    if not pcs:
        return None
    min_len = min(pc.shape[0] for pc in pcs)
    if min_len < 2:
        return None
    stacked = np.stack([pc[:min_len] for pc in pcs], axis=0)
    return stacked.mean(axis=0)


# ═══════════════════════════════════════════════════════════════════
#  7. Direction-arrow helpers
# ═══════════════════════════════════════════════════════════════════

def _add_traj_arrow_mid(ax, mean_pc, color, alpha, lw,
                         head_scale: float = 8, n_trim: int = 3):
    T = len(mean_pc)
    if T < n_trim + 2:
        return 0
    i0 = T - 1 - n_trim
    i1 = T - 1
    ax.annotate(
        '', xy=(mean_pc[i1, 0], mean_pc[i1, 1]),
        xytext=(mean_pc[i0, 0], mean_pc[i0, 1]),
        arrowprops=dict(arrowstyle='-|>',
                        color=color, lw=lw,
                        mutation_scale=head_scale,
                        shrinkA=0, shrinkB=0),
        alpha=alpha, zorder=6)
    return n_trim


def _add_segment_end_arrow(ax, seg_pc, color, alpha, lw,
                            head_scale: float = 6, n_span: int = 2):
    T = len(seg_pc)
    if T < n_span + 1:
        return
    i0 = T - 1 - n_span
    i1 = T - 1
    ax.annotate(
        '', xy=(seg_pc[i1, 0], seg_pc[i1, 1]),
        xytext=(seg_pc[i0, 0], seg_pc[i0, 1]),
        arrowprops=dict(arrowstyle='-|>',
                        color=color, lw=lw * 0.8,
                        mutation_scale=head_scale,
                        shrinkA=0, shrinkB=0),
        alpha=alpha, zorder=6)


# ═══════════════════════════════════════════════════════════════════
#  8. Legend builder
# ═══════════════════════════════════════════════════════════════════

def _build_legend_handles(show_tt: List[str] = None,
                          show_ablated: bool = True,
                          ablated_tt: Optional[List[str]] = None):
    if show_tt is None:
        show_tt = list(TRIAL_TYPES)
    if ablated_tt is None:
        ablated_set = set(show_tt) if show_ablated else set()
    else:
        ablated_set = set(ablated_tt)
    has_any_ablated = bool(ablated_set & set(show_tt))
    col1, col2 = [], []
    for tt in show_tt:
        col1.append(plt.Line2D(
            [], [], color=TRIAL_COLORS[tt], lw=3.0, linestyle='-',
            alpha=0.95, label=f'{TRIAL_SHORT[tt]} intact'))
        if has_any_ablated:
            if tt in ablated_set:
                col2.append(plt.Line2D(
                    [], [], color=TRIAL_COLORS[tt], lw=2.4, linestyle='-',
                    alpha=0.60, label=f'{TRIAL_SHORT[tt]} {ERROR_DISPLAY}'))
            else:
                col2.append(plt.Line2D(
                    [], [], linestyle='none', color='none', label=' '))
    if not has_any_ablated:
        return col1
    handles = []
    for h1, h2 in zip(col1, col2):
        handles.extend([h1, h2])
    return handles


# ═══════════════════════════════════════════════════════════════════
#  Extend trajectories by N autonomous (x=0) steps
# ═══════════════════════════════════════════════════════════════════

def _extend_trajectories_autonomous(
    model: ContinuousTimeRNN,
    trajs_list: List[np.ndarray],
    n_extra_steps: int = EXTEND_N_STEPS,
    device: str = 'cpu',
    batch_size: int = 256,
) -> List[np.ndarray]:
    if not trajs_list:
        return []
    model.eval()
    extended: List[np.ndarray] = []

    for i0 in range(0, len(trajs_list), batch_size):
        i1 = min(i0 + batch_size, len(trajs_list))
        batch_arrs = trajs_list[i0:i1]

        h_finals = np.stack([arr[-1] for arr in batch_arrs])
        h = torch.tensor(h_finals, dtype=torch.float32, device=device)

        extra_parts: List[np.ndarray] = []
        with torch.no_grad():
            for _ in range(n_extra_steps):
                h = model.autonomous_step(h)
                extra_parts.append(h.cpu().numpy())

        extra_np = np.stack(extra_parts, axis=1)

        for b, arr in enumerate(batch_arrs):
            full = np.concatenate([arr, extra_np[b]], axis=0)
            extended.append(full)

    return extended


def compute_all_flow_fields(
    model: ContinuousTimeRNN,
    pca: PCA,
    neuron_char: NeuronCharacterization,
    device: str,
    pc_range: Tuple,
    grid_res: int = GRID_RES_DEFAULT,
    traj_cloud_pc: Optional[np.ndarray] = None,
    traj_cloud_h:  Optional[np.ndarray] = None,
) -> Dict[str, Dict]:
    H = model.hidden_dim
    conds = build_neuron_ablation_conditions(neuron_char, H, device)
    flows: Dict[str, Dict] = {}
    total = len(conds)
    for i, (cn, (afn, adir)) in enumerate(conds.items(), 1):
        label = ABLATION_ROWS.get(cn, cn)
        print(f"    [{i}/{total}] flow: {cn} ({label})  x = 0")
        if cn == 'intact':
            afn_use, adir_use = None, None
        else:
            afn_use, adir_use = afn, adir
        flows[cn] = compute_flow_field(
            model, pca, device, pc_range, grid_res,
            ablation_fn=afn_use,
            ablation_dir=adir_use,
            compute_quiver=(cn == 'intact'),
            traj_cloud_pc=traj_cloud_pc,
            traj_cloud_h=traj_cloud_h)
    return flows


# ═══════════════════════════════════════════════════════════════════
#  2-D panel with extended-tail rendering
# ═══════════════════════════════════════════════════════════════════

def _plot_2d_extended(
    ax, flow: Dict,
    trajs_intact: Dict,
    trajs_abl_ext: Dict,
    pca: PCA,
    highlight_tt: List[str],
    centers: Dict,
    title: str,
    timing: Dict,
    original_total_len: int,
    n_extra_steps: int = EXTEND_N_STEPS,
    show_legend: bool = True,
    n_individual: int = N_INDIVIDUAL_SHOW,
    contour_levels: Optional[np.ndarray] = None,
    show_end_idx: Optional[int] = None,
):
    init_s  = timing['init_steps']
    delay_s = timing['delay_steps']
    Xi, Yi, Sc = flow['Xi'], flow['Yi'], flow['Sc']

    # ── background contour ──
    if contour_levels is not None:
        cf = ax.contourf(Xi, Yi, Sc, levels=contour_levels,
                         cmap=CMAP_LANDSCAPE, extend='neither')
        ax.contour(Xi, Yi, Sc, levels=contour_levels[::2],
                   colors='white', alpha=0.25, linewidths=0.4)
    else:
        cf = ax.contourf(Xi, Yi, Sc, levels=25, cmap=CMAP_LANDSCAPE)
        ax.contour(Xi, Yi, Sc, levels=12,
                   colors='white', alpha=0.25, linewidths=0.4)

    # ── quiver: speed-proportional arrows ──
    if 'Xq' in flow and flow.get('U_dir') is not None:
        _draw_speed_quiver(ax, flow['Xq'], flow['Yq'],
                           flow['U_dir'], flow['V_dir'])

    # ── trajectories ──
    T_orig = original_total_len
    show_truncated = show_end_idx is not None
    ablated_plotted: List[str] = []

    for tt in highlight_tt:
        color = TRIAL_COLORS[tt]

        # intact mean
        intact_arrs = trajs_intact.get(tt, [])
        mean_intact = _mean_traj_pc(intact_arrs, pca, start_idx=0,
                                     end_idx=show_end_idx)
        if mean_intact is not None:
            _draw_phased_trajectory(ax, mean_intact, color,
                                    init_s, delay_s,
                                    base_lw=3.0, base_alpha=0.95)

        # individual extended error trajectories
        ext_arrs = trajs_abl_ext.get(tt, [])
        n_show = min(len(ext_arrs), n_individual)

        for arr_i in ext_arrs[:n_show]:
            if show_truncated:
                arr_slice = arr_i[:show_end_idx]
                pc_i = pca.transform(arr_slice)[:, :2]
                ax.plot(pc_i[:, 0], pc_i[:, 1],
                        color=color, lw=0.45, alpha=0.13, zorder=2.5,
                        solid_capstyle='round')
                ax.scatter(pc_i[-1, 0], pc_i[-1, 1],
                           c=color, s=10, alpha=0.30,
                           edgecolors='none', zorder=2.8)
            else:
                pc_i = pca.transform(arr_i)[:, :2]
                t_cut = min(T_orig, len(pc_i))

                ax.plot(pc_i[:t_cut, 0], pc_i[:t_cut, 1],
                        color=color, lw=0.45, alpha=0.13, zorder=2.5,
                        solid_capstyle='round')

                if len(pc_i) > t_cut:
                    ax.plot(pc_i[t_cut - 1:, 0], pc_i[t_cut - 1:, 1],
                            color=EXTEND_COLOR, lw=EXTEND_LW_IND,
                            alpha=EXTEND_ALPHA_IND, zorder=2.6,
                            solid_capstyle='round')
                    ax.scatter(pc_i[-1, 0], pc_i[-1, 1],
                               c=EXTEND_COLOR, s=12, alpha=0.35,
                               edgecolors='none', zorder=2.8)

        # mean — original portion (phased drawing)
        if ext_arrs:
            mean_orig = _mean_traj_pc(ext_arrs, pca, start_idx=0,
                                       end_idx=(show_end_idx if show_truncated
                                                else T_orig))
            if mean_orig is not None:
                _draw_phased_trajectory(ax, mean_orig, color,
                                        init_s, delay_s,
                                        base_lw=2.4, base_alpha=0.60)
                ablated_plotted.append(tt)

            if not show_truncated:
                mean_ext = _mean_traj_pc(ext_arrs, pca,
                                          start_idx=max(0, T_orig - 1))
                if mean_ext is not None and len(mean_ext) > 2:
                    ax.plot(mean_ext[:, 0], mean_ext[:, 1],
                            color='white', lw=EXTEND_LW_MEAN + 2.0,
                            solid_capstyle='round', alpha=0.45, zorder=3)
                    ax.plot(mean_ext[:, 0], mean_ext[:, 1],
                            color=EXTEND_COLOR, lw=EXTEND_LW_MEAN,
                            solid_capstyle='round',
                            alpha=EXTEND_ALPHA_MEAN, zorder=4)
                    if len(mean_ext) > 4:
                        _add_traj_arrow_mid(ax, mean_ext, EXTEND_COLOR,
                                            alpha=EXTEND_ALPHA_MEAN,
                                            lw=EXTEND_LW_MEAN,
                                            head_scale=9, n_trim=3)
                    ax.scatter(mean_ext[-1, 0], mean_ext[-1, 1],
                               c=EXTEND_COLOR, s=90, marker='*',
                               edgecolors='k', linewidths=0.8, zorder=7)

    # ── attractor centres ──
    _ac = {'NREM': '#27ae60', 'Wake': '#f39c12'}
    for name, (pc1c, pc2c) in centers.items():
        col = _ac.get(name, '#9b59b6')
        ax.scatter(pc1c, pc2c, c=col, s=130, marker='o',
                   edgecolors='k', linewidths=1.2, zorder=7)
        ax.text(pc1c, pc2c + 0.3, name, color='black', fontsize=7,
                fontweight='bold', ha='center', va='bottom', zorder=8)

    ax.set_xlabel('PC1', fontsize=8)
    ax.set_ylabel('PC2', fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)

    if show_legend:
        handles = _build_legend_handles(
            show_tt=highlight_tt, show_ablated=True,
            ablated_tt=ablated_plotted)
        if not show_truncated:
            handles.append(plt.Line2D(
                [], [], color=EXTEND_COLOR, lw=2.5,
                alpha=EXTEND_ALPHA_MEAN,
                label=f'+{n_extra_steps} auton. steps'))
            handles.append(plt.Line2D(
                [], [], color=EXTEND_COLOR, marker='*', markersize=8,
                linestyle='none', label='Extended endpoint'))
        leg = ax.legend(
            handles=handles, loc='upper right', fontsize=5, ncol=2,
            framealpha=0.92, edgecolor='gray', fancybox=True,
            borderpad=0.4, handlelength=1.2, labelspacing=0.25,
            columnspacing=0.8, handletextpad=0.4)
        leg.set_zorder(10)
    return cf


# ═══════════════════════════════════════════════════════════════════
#  Generate the extended-trajectory figure
# ═══════════════════════════════════════════════════════════════════

def generate_extended_trajectory_figure(
    seed: int,
    window: str,
    config: ExperimentConfig,
    model: ContinuousTimeRNN,
    flows_auto: Dict[str, Dict],
    flows_input_cond: Dict[str, Dict],
    pca: PCA,
    centers: Dict,
    timing: Dict,
    trajs: Dict,
    pred_states: Dict,
    device: str,
    n_extra_steps: int = EXTEND_N_STEPS,
    min_errors: int = 1,
) -> Tuple[List[str], List[str]]:
    set_publication_style()
    plottable, skipped = _plottable_conditions(
        pred_states, min_errors=min_errors)
    if not plottable:
        print(f"  ⚠ seed {seed}, window {window}: nothing plottable "
              f"for extended-trajectory figure — skipped")
        return [], list(ABLATION_ROWS.keys())

    total_s = timing['total_steps']
    original_total_len = total_s + 1

    # ── extend error trajectories ──
    print(f"    Extending error trajectories by "
          f"{n_extra_steps} autonomous steps ...")
    ext_trajs: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for cond in plottable:
        hl_tts = ROW_HIGHLIGHT.get(cond, [])
        raw    = trajs.get(cond, {})
        preds  = pred_states.get(cond, {})
        ext_trajs[cond] = {}
        for tt in hl_tts:
            arrs = raw.get(tt, [])
            ps   = preds.get(tt, [])
            err_arrs, _, _ = filter_error_trials(arrs, ps, tt)
            if err_arrs:
                ext = _extend_trajectories_autonomous(
                    model, err_arrs,
                    n_extra_steps=n_extra_steps,
                    device=device)
                ext_trajs[cond][tt] = ext
                print(f"      {cond}/{tt}: {len(ext)} trajs  "
                      f"T={original_total_len} → "
                      f"{original_total_len + n_extra_steps}")
            else:
                ext_trajs[cond][tt] = []

    # ── figure ──
    n_rows = len(plottable)
    n_main_cols = len(INPUT_COL_ORDER)
    panel_w, panel_h = 4.2, 4.5
    fig_w = n_main_cols * panel_w + 0.8
    fig_h = n_rows * panel_h + 0.8

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(
        n_rows, n_main_cols + 1,
        width_ratios=[1] * n_main_cols + [0.04],
        hspace=0.35, wspace=0.25, figure=fig)

    def _get_flow(col_key):
        if col_key == 'x0':
            return flows_auto.get('intact')
        return flows_input_cond.get(col_key)

    for ri, cond in enumerate(plottable):
        label    = ABLATION_ROWS[cond]
        hl_tt    = ROW_HIGHLIGHT.get(cond, TRIAL_TYPES[:2])
        act_cols = ROW_ACTIVE_INPUT_COLS.get(cond, ['x0'])

        trajs_intact_all = trajs.get('intact', {})
        trajs_ext        = ext_trajs.get(cond, {})

        sc_parts = []
        for ck in act_cols:
            fl = _get_flow(ck)
            if fl is not None:
                sc_parts.append(fl['Sc'].ravel())
        if sc_parts:
            a = np.concatenate(sc_parts)
            m = (a.max() - a.min()) * 0.02
            contour_levels = np.linspace(a.min() - m, a.max() + m, 26)
        else:
            contour_levels = None

        last_cf = None
        for ci, col_key in enumerate(INPUT_COL_ORDER):
            ax = fig.add_subplot(gs[ri, ci])
            if col_key not in act_cols:
                ax.set_visible(False)
                continue
            flow = _get_flow(col_key)
            if flow is None:
                ax.set_visible(False)
                continue

            title_str = f'{label}\n{INPUT_COL_LABELS[col_key]}'

            if col_key != 'x0':
                show_end = timing['init_steps'] + 1
            else:
                show_end = None

            cf = _plot_2d_extended(
                ax, flow,
                trajs_intact_all, trajs_ext,
                pca, hl_tt, centers,
                title=title_str, timing=timing,
                original_total_len=original_total_len,
                n_extra_steps=n_extra_steps,
                show_legend=(ci == 0),
                contour_levels=contour_levels,
                show_end_idx=show_end)
            last_cf = cf

        cax = fig.add_subplot(gs[ri, n_main_cols])
        if last_cf is not None:
            cb = fig.colorbar(last_cf, cax=cax)
            cb.set_label('log velocity', fontsize=7)
            cb.ax.tick_params(labelsize=6)
        else:
            cax.set_visible(False)

    fig.suptitle(
        f'Extended Trajectories (+{n_extra_steps} autonomous steps)  │  '
        f'Seed {seed}  │  {window}',
        fontsize=FONT['suptitle'], fontweight='bold', y=1.01)

    save_path = os.path.join(
        _fig_dir(config),
        f'flow_extended_traj_seed{seed}_{window}.png')
    save_figure(fig, save_path)
    plt.close(fig)
    print(f"  ✓ Extended trajectory figure: {save_path}  "
          f"({len(plottable)}/{len(ABLATION_ROWS)} rows)")
    return plottable, skipped


# ═══════════════════════════════════════════════════════════════════
#  Phased trajectory drawer
# ═══════════════════════════════════════════════════════════════════

def _draw_phased_trajectory(ax, mean_pc: np.ndarray, color: str,
                            init_s: int, delay_s: int,
                            base_lw: float = 3.0,
                            base_alpha: float = 0.95):
    T = len(mean_pc)
    if T < 3:
        return
    segments = [
        ('init',     0,                          min(init_s + 1, T)),
        ('delay',    max(0, init_s),              min(init_s + delay_s + 1, T)),
        ('response', max(0, init_s + delay_s),    T),
    ]
    for phase_name, s_start, s_end in segments:
        seg = mean_pc[s_start:s_end]
        if len(seg) < 2:
            continue
        props = _PHASE_PROPS[phase_name]
        lw    = base_lw    * props['lw_f']
        alpha = base_alpha * props['a_f']
        pc    = _phase_adjusted_color(color, phase_name)
        is_last = (phase_name == 'response')
        n_trim  = 3 if (is_last and len(seg) > 5) else 0
        if n_trim > 0:
            plot_seg = seg[:len(seg) - n_trim]
        else:
            plot_seg = seg
        ax.plot(plot_seg[:, 0], plot_seg[:, 1],
                color='white', lw=lw + 2.0,
                solid_capstyle='round', alpha=alpha * 0.55, zorder=3)
        ax.plot(plot_seg[:, 0], plot_seg[:, 1],
                color=pc, lw=lw, linestyle=props['ls'],
                solid_capstyle='round', alpha=alpha, zorder=4)
        if is_last and n_trim > 0:
            _add_traj_arrow_mid(ax, seg, pc, alpha=alpha,
                                lw=lw, head_scale=props['arrow_scale'],
                                n_trim=n_trim)
        elif len(seg) >= 3:
            _add_segment_end_arrow(ax, seg, pc, alpha=alpha,
                                   lw=lw,
                                   head_scale=props['arrow_scale'])


# ═══════════════════════════════════════════════════════════════════
#  9. 2-D flow-field panel
# ═══════════════════════════════════════════════════════════════════

def _plot_2d(
    ax, flow: Dict,
    trajs_intact: Dict,
    trajs_ablated: Dict,
    pca: PCA,
    highlight_tt: List[str],
    centers: Dict,
    title: str,
    timing: Dict,
    show_legend: bool = True,
    n_individual: int = N_INDIVIDUAL_SHOW,
    contour_levels: Optional[np.ndarray] = None,
    show_end_idx: Optional[int] = None,
):
    init_s  = timing['init_steps']
    delay_s = timing['delay_steps']
    Xi = flow['Xi']
    Yi = flow['Yi']
    Sc = flow['Sc']

    if contour_levels is not None:
        cf = ax.contourf(Xi, Yi, Sc, levels=contour_levels,
                         cmap=CMAP_LANDSCAPE, extend='neither')
        ax.contour(Xi, Yi, Sc, levels=contour_levels[::2],
                   colors='white', alpha=0.25, linewidths=0.4)
    else:
        cf = ax.contourf(Xi, Yi, Sc, levels=25, cmap=CMAP_LANDSCAPE)
        ax.contour(Xi, Yi, Sc, levels=12,
                   colors='white', alpha=0.25, linewidths=0.4)

    # ── quiver: speed-proportional arrows ──
    if 'Xq' in flow and flow.get('U_dir') is not None:
        _draw_speed_quiver(ax, flow['Xq'], flow['Yq'],
                           flow['U_dir'], flow['V_dir'])

    ablated_plotted: List[str] = []
    for tt in highlight_tt:
        color = TRIAL_COLORS[tt]
        intact_arrs = trajs_intact.get(tt, [])
        mean_intact = _mean_traj_pc(intact_arrs, pca, start_idx=0,
                                     end_idx=show_end_idx)
        if mean_intact is not None:
            _draw_phased_trajectory(ax, mean_intact, color,
                                    init_s, delay_s,
                                    base_lw=3.0, base_alpha=0.95)
        abl_arrs = trajs_ablated.get(tt, [])
        n_show = min(len(abl_arrs), n_individual)
        for arr_i in abl_arrs[:n_show]:
            arr_slice = arr_i[:show_end_idx] if show_end_idx is not None else arr_i
            pc_i = pca.transform(arr_slice)[:, :2]
            ax.plot(pc_i[:, 0], pc_i[:, 1],
                    color=color, lw=0.45, alpha=0.13, zorder=2.5,
                    solid_capstyle='round')
            ax.scatter(pc_i[-1, 0], pc_i[-1, 1],
                       c=color, s=10, alpha=0.30,
                       edgecolors='none', zorder=2.8)
        mean_abl = _mean_traj_pc(abl_arrs, pca, start_idx=0,
                                  end_idx=show_end_idx)
        if mean_abl is not None:
            _draw_phased_trajectory(ax, mean_abl, color,
                                    init_s, delay_s,
                                    base_lw=2.4, base_alpha=0.60)
            ablated_plotted.append(tt)

    _ac = {'NREM': '#27ae60', 'Wake': '#f39c12'}
    for name, (pc1c, pc2c) in centers.items():
        col = _ac.get(name, '#9b59b6')
        ax.scatter(pc1c, pc2c, c=col, s=130, marker='o',
                   edgecolors='k', linewidths=1.2, zorder=7)
        ax.text(pc1c, pc2c + 0.3, name,
                color='black', fontsize=7, fontweight='bold',
                ha='center', va='bottom', zorder=8)

    ax.set_xlabel('PC1', fontsize=8)
    ax.set_ylabel('PC2', fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)

    if show_legend:
        handles = _build_legend_handles(show_tt=highlight_tt,
                                         show_ablated=True,
                                         ablated_tt=ablated_plotted)
        leg = ax.legend(
            handles=handles, loc='upper right',
            fontsize=5, ncol=2,
            framealpha=0.92, edgecolor='gray', fancybox=True,
            borderpad=0.4, handlelength=1.2, labelspacing=0.25,
            columnspacing=0.8, handletextpad=0.4)
        leg.set_zorder(10)
    return cf


# ═══════════════════════════════════════════════════════════════════
#  11. Input-conditioned multi-column figure
# ═══════════════════════════════════════════════════════════════════

def generate_input_conditioned_figure(
    seed: int,
    window: str,
    config: ExperimentConfig,
    flows_auto: Dict[str, Dict],
    flows_input_cond: Dict[str, Dict],
    pca: PCA,
    centers: Dict,
    timing: Dict,
    trajs: Dict,
    pred_states: Dict,
    min_errors: int = 1,
) -> Tuple[List[str], List[str]]:
    set_publication_style()
    plottable, skipped = _plottable_conditions(
        pred_states, min_errors=min_errors)
    if not plottable:
        print(f"  ⚠ seed {seed}, window {window}: no plottable conditions "
              f"for input-conditioned figure — skipped")
        return [], list(ABLATION_ROWS.keys())

    n_rows = len(plottable)
    n_main_cols = len(INPUT_COL_ORDER)
    panel_w, panel_h = 4.2, 4.5
    fig_w = n_main_cols * panel_w + 0.8
    fig_h = n_rows * panel_h + 0.8

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(
        n_rows, n_main_cols + 1,
        width_ratios=[1] * n_main_cols + [0.04],
        hspace=0.35, wspace=0.25, figure=fig)

    def _get_flow(col_key):
        if col_key == 'x0':
            return flows_auto.get('intact')
        return flows_input_cond.get(col_key)

    for ri, cond in enumerate(plottable):
        label = ABLATION_ROWS[cond]
        hl_tt = ROW_HIGHLIGHT.get(cond, TRIAL_TYPES[:2])
        active_cols = ROW_ACTIVE_INPUT_COLS.get(cond, ['x0'])

        trajs_intact_all = trajs.get('intact', {})
        trajs_abl_raw = trajs.get(cond, {})
        preds_cond    = pred_states.get(cond, {})
        trajs_abl_filtered: Dict[str, List[np.ndarray]] = {}
        for tt in hl_tt:
            arrs  = trajs_abl_raw.get(tt, [])
            preds = preds_cond.get(tt, [])
            kept, _, _ = filter_error_trials(arrs, preds, tt)
            trajs_abl_filtered[tt] = kept

        sc_parts = []
        for col_key in active_cols:
            fl = _get_flow(col_key)
            if fl is not None:
                sc_parts.append(fl['Sc'].ravel())
        if sc_parts:
            all_sc = np.concatenate(sc_parts)
            sc_lo = all_sc.min()
            sc_hi = all_sc.max()
            margin = (sc_hi - sc_lo) * 0.02
            contour_levels = np.linspace(sc_lo - margin, sc_hi + margin, 26)
        else:
            contour_levels = None

        last_cf = None
        for ci, col_key in enumerate(INPUT_COL_ORDER):
            ax = fig.add_subplot(gs[ri, ci])
            if col_key not in active_cols:
                ax.set_visible(False)
                continue
            flow = _get_flow(col_key)
            if flow is None:
                ax.set_visible(False)
                continue
            title_str = f'{label}\n{INPUT_COL_LABELS[col_key]}'

            if col_key != 'x0':
                show_end = timing['init_steps'] + 1
            else:
                show_end = None

            cf = _plot_2d(
                ax, flow,
                trajs_intact_all, trajs_abl_filtered,
                pca, hl_tt, centers,
                title=title_str, timing=timing,
                show_legend=(ci == 0),
                contour_levels=contour_levels,
                show_end_idx=show_end)
            last_cf = cf

        cax = fig.add_subplot(gs[ri, n_main_cols])
        if last_cf is not None:
            cbar = fig.colorbar(last_cf, cax=cax)
            cbar.set_label('log velocity', fontsize=7)
            cbar.ax.tick_params(labelsize=6)
        else:
            cax.set_visible(False)

    fig.suptitle(
        f'Input-Conditioned Flow Landscapes  │  Seed {seed}  │  {window}',
        fontsize=FONT['suptitle'], fontweight='bold', y=1.01)
    save_path = os.path.join(
        _fig_dir(config),
        f'flow_input_cond_seed{seed}_{window}.png')
    save_figure(fig, save_path)
    plt.close(fig)
    print(f"  ✓ Input-conditioned figure: {save_path}  "
          f"({len(plottable)}/{len(ABLATION_ROWS)} rows)")
    return plottable, skipped


# ═══════════════════════════════════════════════════════════════════
#  12. Ablation activity heatmaps
# ═══════════════════════════════════════════════════════════════════

ZONE = {
    'init_bg':     '#F4F6F7',
    'delay_bg':    '#FDEBD0',
    'nrem_bg':     '#FDF2E9',
    'wake_bg':     '#EBF5FB',
    'chance':      '#ABB2B9',
    'edge':        '#2C3E50',
}
CMAP_HEATMAP = 'YlOrRd'

ABLATION_TO_GROUP_KEY = {
    'ablate_n2w':  'n2w',
    'ablate_w2n':  'w2n',
    'ablate_both': 'both_trans',
}
ABLATION_GROUP_LABEL = {
    'ablate_n2w':  'N→W trans',
    'ablate_w2n':  'W→N trans',
    'ablate_both': 'both trans',
}
ABLATION_OVERLAY_RGBA = (0.55, 0.0, 0.90, 0.35)
ABLATION_BORDER_COLOR = '#7D3C98'


def _auto_cmap_range(arr):
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99))
    if lo >= -0.01:
        return CMAP_HEATMAP, 0.0, hi
    vabs = max(abs(lo), abs(hi))
    return 'RdBu_r', -vabs, vabs


def _draw_hs_panel(ax, data, init, delay, cmap, vmin, vmax,
                   draw_diag=False):
    N, T = data.shape
    resp_start = init + delay
    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest', origin='upper')
    if draw_diag:
        ax.plot([0, T - 1], [0, N - 1], color='white', ls=':', lw=0.6,
                alpha=0.5)
    ax.axvline(init - 0.5, color=ZONE['edge'], ls='--', lw=0.8, alpha=0.7)
    if delay > 0:
        ax.axvline(resp_start - 0.5, color=ZONE['edge'], ls='--', lw=0.8,
                   alpha=0.5)
    step = max(1, T // 5)
    ax.set_xticks(range(0, T, step))
    yt = [i for i in [0, N // 4, N // 2, 3 * N // 4, N - 1] if i < N]
    ax.set_yticks(yt)
    return im


def _build_ablation_overlay(
    abl_name, neuron_char, H, T, init_s, delay_s, order,
    ablation_window='init_delay',
):
    group_key = ABLATION_TO_GROUP_KEY.get(abl_name)
    if group_key is None:
        return None, 0, 0, ''
    ids = _resolve_neuron_ids(neuron_char, group_key)
    affected_set = set(int(i) for i in ids)
    if len(affected_set) == 0:
        return None, 0, 0, ''
    abl_end = (T if ablation_window == 'full_trial'
               else min(init_s + delay_s, T))
    mask = np.zeros((H, T), dtype=bool)
    n_aff = 0
    for r in range(H):
        if int(order[r]) in affected_set:
            mask[r, :abl_end] = True
            n_aff += 1
    if n_aff == 0:
        return None, abl_end, 0, ''
    overlay = np.zeros((H, T, 4), dtype=np.float32)
    overlay[mask] = ABLATION_OVERLAY_RGBA
    group_label = ABLATION_GROUP_LABEL.get(abl_name, group_key)
    return overlay, abl_end, n_aff, group_label


def plot_ablation_activity_heatmaps(
    trajs, timing, save_dir=None, seed=0,
    neuron_char=None, ablation_window='init_delay',
):
    set_publication_style()
    TT_LAB_MAP = {
        'nrem_to_wake': 'N→W', 'wake_to_nrem': 'W→N',
        'nrem_only': 'N Only', 'wake_only': 'W Only',
    }
    init_s = timing['init_steps']
    delay_s = timing['delay_steps']

    def _mean(arrs):
        if not arrs:
            return None
        L = min(a.shape[0] - 1 for a in arrs)
        if L < 1:
            return None
        return np.stack([a[1:L + 1] for a in arrs]).mean(0)

    ma = {cn: {tt: _mean(arrs) for tt, arrs in cd.items()}
          for cn, cd in trajs.items()}

    figs = []
    for abl_name, abl_label in ABLATION_ROWS.items():
        if abl_name not in ma:
            continue
        show_tt = ROW_HIGHLIGHT.get(abl_name, [])
        if not show_tt:
            continue
        show_lab = [TT_LAB_MAP[tt] for tt in show_tt]
        n_rows = len(show_tt)
        gv = []
        for tt in show_tt:
            for cn in ('intact', abl_name):
                d = ma.get(cn, {}).get(tt)
                if d is not None:
                    gv.append(d.ravel())
        if not gv:
            continue
        cmap_act, vmin_act, vmax_act = _auto_cmap_range(np.concatenate(gv))
        dv = []
        for tt in show_tt:
            i_d = ma.get('intact', {}).get(tt)
            a_d = ma.get(abl_name, {}).get(tt)
            if i_d is not None and a_d is not None:
                L = min(i_d.shape[0], a_d.shape[0])
                dv.append((a_d[:L] - i_d[:L]).ravel())
        if dv:
            dv_all = np.concatenate(dv)
            dv_max = max(abs(dv_all.min()), abs(dv_all.max()), 1e-8)
            vmin_d, vmax_d = -dv_max, dv_max
        else:
            vmin_d, vmax_d = -1, 1
        cmap_diff = 'RdBu_r'
        pw, ph = 4.0, 6.0
        fig = plt.figure(figsize=(3 * pw + 2.0, n_rows * ph + 1.2))
        gs_hm = gridspec.GridSpec(
            n_rows, 6,
            width_ratios=[1, 1, 0.1, 0.1, 1.0, 0.1],
            hspace=0.38, wspace=0.4, figure=fig)
        for ri, (tt, lab) in enumerate(zip(show_tt, show_lab)):
            ax_i   = fig.add_subplot(gs_hm[ri, 0])
            ax_a   = fig.add_subplot(gs_hm[ri, 1])
            ax_cb1 = fig.add_subplot(gs_hm[ri, 2])
            ax_d   = fig.add_subplot(gs_hm[ri, 4])
            ax_cb2 = fig.add_subplot(gs_hm[ri, 5])
            int_d = ma.get('intact', {}).get(tt)
            abl_d = ma.get(abl_name, {}).get(tt)
            if int_d is None:
                for ax in (ax_i, ax_a, ax_d):
                    ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                            ha='center', va='center',
                            fontsize=FONT['title'], color='gray')
                    ax.set_xticks([]); ax.set_yticks([])
                ax_cb1.set_visible(False); ax_cb2.set_visible(False)
                continue
            int_ht = int_d.T
            H, T = int_ht.shape
            order = np.argsort(np.argmax(int_ht, axis=1))
            im_act = _draw_hs_panel(ax_i, int_ht[order], init_s, delay_s,
                                    cmap_act, vmin_act, vmax_act,
                                    draw_diag=True)
            if abl_d is not None:
                abl_ht = abl_d.T
                T_abl = abl_ht.shape[1]
                if T_abl < T:
                    pad = np.tile(abl_ht[:, -1:], (1, T - T_abl))
                    abl_ht = np.concatenate([abl_ht, pad], axis=1)
                elif T_abl > T:
                    abl_ht = abl_ht[:, :T]
                _draw_hs_panel(ax_a, abl_ht[order], init_s, delay_s,
                               cmap_act, vmin_act, vmax_act, draw_diag=True)
                if neuron_char is not None:
                    overlay, abl_end_col, n_aff, grp_label = \
                        _build_ablation_overlay(
                            abl_name, neuron_char, H, T,
                            init_s, delay_s, order, ablation_window)
                    if overlay is not None:
                        ax_a.imshow(overlay, aspect='auto',
                                    interpolation='nearest', origin='upper')
                        affected_rows = np.where(overlay[:, 0, 3] > 0)[0]
                        if len(affected_rows) > 0:
                            splits = np.where(
                                np.diff(affected_rows) > 1)[0] + 1
                            groups = np.split(affected_rows, splits)
                            for grp in groups:
                                if len(grp) == 0:
                                    continue
                                rect = Rectangle(
                                    (-0.5, grp[0] - 0.5),
                                    abl_end_col, len(grp),
                                    linewidth=1.8,
                                    edgecolor=ABLATION_BORDER_COLOR,
                                    facecolor='none', linestyle='--',
                                    zorder=10)
                                ax_a.add_patch(rect)
                        if abl_end_col < T:
                            ax_a.axvline(abl_end_col - 0.5,
                                         color=ABLATION_BORDER_COLOR,
                                         ls=':', lw=1.8, alpha=0.85,
                                         zorder=9)
                        ax_a.text(
                            0.02, 0.98,
                            f'▮ Silenced: {n_aff} {grp_label} neurons',
                            transform=ax_a.transAxes,
                            fontsize=6, fontweight='bold',
                            color=ABLATION_BORDER_COLOR,
                            va='top', ha='left',
                            bbox=dict(boxstyle='round,pad=0.2',
                                      facecolor='white', alpha=0.88,
                                      edgecolor=ABLATION_BORDER_COLOR,
                                      linewidth=0.8),
                            zorder=11)
            else:
                ax_a.text(0.5, 0.5, 'N/A', transform=ax_a.transAxes,
                          ha='center', va='center',
                          fontsize=FONT['title'], color='gray')
                ax_a.set_xticks([]); ax_a.set_yticks([])
            im_diff = None
            if abl_d is not None:
                L = min(int_d.shape[0], abl_d.shape[0])
                diff_ht = (abl_d[:L] - int_d[:L]).T
                if diff_ht.shape[1] < T:
                    diff_ht = np.concatenate(
                        [diff_ht,
                         np.zeros((H, T - diff_ht.shape[1]))], axis=1)
                im_diff = _draw_hs_panel(ax_d, diff_ht[order], init_s,
                                         delay_s, cmap_diff, vmin_d,
                                         vmax_d, draw_diag=False)
            else:
                ax_d.text(0.5, 0.5, 'N/A', transform=ax_d.transAxes,
                          ha='center', va='center',
                          fontsize=FONT['title'], color='gray')
                ax_d.set_xticks([]); ax_d.set_yticks([])
            for ax in (ax_i, ax_a, ax_d):
                for sp in ax.spines.values():
                    sp.set_linewidth(2.5)
            ax_i.set_ylabel(lab, fontsize=FONT['label'], fontweight='bold',
                            color=TRIAL_COLORS[tt])
            if ri == 0:
                ax_i.set_title('Intact', fontsize=FONT['title'],
                               fontweight='bold')
                ax_a.set_title(abl_label, fontsize=FONT['title'],
                               fontweight='bold')
                ax_d.set_title('Δ (Abl − Int)', fontsize=FONT['title'],
                               fontweight='bold')
            if ri < n_rows - 1:
                for ax in (ax_i, ax_a, ax_d):
                    ax.set_xticklabels([])
            else:
                for ax in (ax_i, ax_a, ax_d):
                    ax.set_xlabel('Time step', fontsize=FONT['label'])
            cb1 = fig.colorbar(im_act, cax=ax_cb1)
            cb1.ax.tick_params(labelsize=5)
            if ri == n_rows - 1:
                cb1.set_label('Act.', fontsize=6)
            if im_diff is not None:
                cb2 = fig.colorbar(im_diff, cax=ax_cb2)
                cb2.ax.tick_params(labelsize=5)
                if ri == n_rows - 1:
                    cb2.set_label('Δ Act.', fontsize=6)
            else:
                ax_cb2.set_visible(False)
        tt_str = ' + '.join(show_lab)
        fig.suptitle(
            f'Neural Activity: Intact vs {abl_label}  │  {tt_str}  │  '
            f'Seed {seed}',
            fontsize=FONT['suptitle'], fontweight='bold', y=1.01)
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            p = os.path.join(
                save_dir,
                f'ablation_activity_{abl_name}_seed{seed}.png')
            save_figure(fig, p)
            print(f'  ✓ {p}')
        plt.close(fig)
        figs.append(fig)
    return figs

def plot_4_types_heatmaps(trajs, save_dir=None, seed=0):
    set_publication_style()

    TT_ORDER  = ['nrem_to_wake', 'wake_to_nrem', 'wake_only', 'nrem_only']
    TT_LABELS = ['N→W', 'W→N', 'W→W', 'N→N']

    intact = None
    for wn_data in trajs.values():
        if 'intact' in wn_data:
            intact = wn_data['intact']
            break
    if intact is None:
        print(' No intact trajectories found; skipping.')
        return None

    def _trial_mean(arrs):
        if not arrs:
            return None
        L = min(a.shape[0] - 1 for a in arrs)
        if L < 1:
            return None
        return np.stack([a[1:L + 1] for a in arrs]).mean(0)

    means = {tt: _trial_mean(intact.get(tt, [])) for tt in TT_ORDER}
    valid = [(tt, lab) for tt, lab in zip(TT_ORDER, TT_LABELS)
             if means[tt] is not None]
    if not valid:
        print(' No valid data; skipping.')
        return None

    all_vals = np.concatenate([means[tt].ravel() for tt, _ in valid])
    vlim = max(abs(all_vals.min()), abs(all_vals.max()), 1e-8)

    n = len(valid)
    fig = plt.figure(figsize=(4.2 * n + 1.0, 6.5))
    gs = gridspec.GridSpec(
        1, n + 1,
        width_ratios=[1] * n + [0.04],
        wspace=0.40, figure=fig,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    cax  = fig.add_subplot(gs[0, n])

    im_ref = None
    for ci, (tt, lab) in enumerate(valid):
        ax  = axes[ci]
        mat = means[tt].T
        H, T = mat.shape

        order      = np.argsort(np.argmax(mat, axis=1))
        mat_sorted = mat[order]

        im = ax.imshow(
            mat_sorted, aspect='auto', interpolation='nearest',
            origin='upper', cmap='RdBu_r', vmin=-vlim, vmax=vlim,
        )
        im_ref = im

        ax.set_title(lab, fontsize=FONT['title'], fontweight='bold')
        ax.set_xlabel('Time step', fontsize=FONT['label'])

        ytick_pos = [i for i in [0, 32, 64, 96, H - 1] if i < H]
        ytick_ids = [str(order[i]) for i in ytick_pos]
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_ids, fontsize=7)
        if ci == 0:
            ax.set_ylabel('Neuron ID (sorted)', fontsize=FONT['label'])

        xticks = [0, 10, 42]
        if T - 1 > 42:
            xticks.append(T - 1)
        ax.set_xticks(xticks)
        ax.axvline(10, color='grey', ls='--', lw=0.8, alpha=0.55)
        ax.axvline(42, color='grey', ls='--', lw=0.8, alpha=0.55)

        for sp in ax.spines.values():
            sp.set_linewidth(2.0)

    cb = fig.colorbar(im_ref, cax=cax)
    cb.ax.tick_params(labelsize=7)
    cb.set_label('Act.', fontsize=8)

    fig.suptitle(
        f'Intact Neural Activity │ Seed {seed}',
        fontsize=FONT.get('suptitle', 14), fontweight='bold',
    )
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'intact_4types_seed{seed}.png')
        save_figure(fig, path)
        print(f'  ✓ {path}')
    plt.close(fig)
    return fig

# ═══════════════════════════════════════════════════════════════════
#  13. Per-seed orchestrator
# ═══════════════════════════════════════════════════════════════════

def analyse_single_seed(
    seed: int,
    model: ContinuousTimeRNN,
    data_loader,
    neuron_char: NeuronCharacterization,
    config: ExperimentConfig,
    device: str,
    windows: List[str] = None,
    grid_res: int = GRID_RES_DEFAULT,
    force_rerun: bool = False,
    min_errors: int = 1,
) -> Dict:
    if windows is None:
        windows = list(DYNAMICS_WINDOWS.keys())
    cdir = _cache_dir(config)

    all_trajs:   Dict[str, Dict] = {}
    all_timings: Dict[str, Dict] = {}
    all_preds:   Dict[str, Dict] = {}

    for wn in windows:
        cache_t = os.path.join(cdir, f'seed{seed}_{wn}_trajs.pkl')
        if not force_rerun:
            cached = _load(cache_t)
            if (cached is not None
                    and isinstance(cached, tuple) and len(cached) == 3):
                all_trajs[wn], all_timings[wn], all_preds[wn] = cached
                print(f"  [{wn}] trajectories loaded from cache")
                continue
        print(f"  [{wn}] collecting trajectories ...")
        t, tm, ps = collect_dynamics_trajectories(
            model, data_loader, neuron_char, device,
            ablation_window=wn)
        all_trajs[wn]   = t
        all_timings[wn] = tm
        all_preds[wn]   = ps
        _save((t, tm, ps), cache_t)

    for wn in windows:
        all_ok, detail = _check_error_coverage(
            all_preds[wn], min_errors=min_errors)
        _print_error_coverage(seed, detail, all_ok, min_errors)

    cache_pca = os.path.join(cdir, f'seed{seed}_pca_intact_trans.pkl')
    pca = None if force_rerun else _load(cache_pca)
    if pca is None:
        print("  Fitting PCA (INTACT) ...")
        first_wn = windows[0]
        combined_intact: Dict = {
            'intact': all_trajs[first_wn].get('intact', {})
        }
        ref_timing = all_timings[first_wn]
        pca = fit_dynamics_pca(combined_intact, ref_timing, n_components=5)
        _save(pca, cache_pca)

    ref_timing = list(all_timings.values())[0]
    combined_for_range: Dict = {}
    for t in all_trajs.values():
        for cn, cn_data in t.items():
            combined_for_range.setdefault(cn, {})
            for tt, arrs in cn_data.items():
                combined_for_range[cn].setdefault(tt, []).extend(arrs)
    pc_range = _get_pc_range(combined_for_range, pca, ref_timing)

    first_wn_trajs = list(all_trajs.values())[0]
    traj_cloud_pc, traj_cloud_h = _build_trajectory_cloud(first_wn_trajs, pca)
    if len(traj_cloud_pc) > 0 and traj_cloud_pc.shape[1] > 2:
        print(f"  Trajectory cloud: {len(traj_cloud_pc)} points  "
            f"(hidden_dim={traj_cloud_h.shape[1]}, "
            f"PC3 ∈ [{traj_cloud_pc[:, 2].min():.2f}, "
            f"{traj_cloud_pc[:, 2].max():.2f}])")
    else:
        print(f"  Trajectory cloud: {len(traj_cloud_pc)} points  "
            f"(hidden_dim={traj_cloud_h.shape[1] if traj_cloud_h.size else 0})")

    ref_trajs = list(all_trajs.values())[0]
    ref_trajs_intact = ref_trajs.get('intact', {})
    centers = _compute_attractor_centers(ref_trajs_intact, pca)

    cache_ff = os.path.join(cdir, f'seed{seed}_flow_fields.pkl')
    flows = None if force_rerun else _load(cache_ff)
    if flows is None:
        print("  Computing zero-input q-landscape flow fields ...")
        flows = compute_all_flow_fields(
            model, pca, neuron_char, device, pc_range,
            grid_res=grid_res,
            traj_cloud_pc=traj_cloud_pc,
            traj_cloud_h=traj_cloud_h)
        _save(flows, cache_ff)

    cache_icf = os.path.join(cdir, f'seed{seed}_input_cond_flows_v2.pkl')
    flows_input_cond = None if force_rerun else _load(cache_icf)
    if flows_input_cond is None:
        print("  Computing input-conditioned flow fields (intact) ...")
        flows_input_cond = compute_input_conditioned_flows(
            model, pca, device, pc_range, grid_res=grid_res,
            traj_cloud_pc=traj_cloud_pc,
            traj_cloud_h=traj_cloud_h)
        _save(flows_input_cond, cache_icf)

    ref_trajs = list(all_trajs.values())[0]
    centers = _compute_attractor_centers(ref_trajs['intact'], pca)

    panel_summary: Dict[str, Tuple[List[str], List[str]]] = {}

    for wn in windows:
        print(f"\n  Generating input-conditioned figure: "
              f"seed={seed}, window={wn}")
        generate_input_conditioned_figure(
            seed=seed, window=wn, config=config,
            flows_auto=flows,
            flows_input_cond=flows_input_cond,
            pca=pca, centers=centers,
            timing=all_timings[wn],
            trajs=all_trajs[wn],
            pred_states=all_preds[wn],
            min_errors=min_errors,
        )
    for wn in windows:
        print(f"\n  Generating extended trajectory figure: "
              f"seed={seed}, window={wn}")
        generate_extended_trajectory_figure(
            seed=seed, window=wn, config=config,
            model=model,
            flows_auto=flows,
            flows_input_cond=flows_input_cond,
            pca=pca, centers=centers,
            timing=all_timings[wn],
            trajs=all_trajs[wn],
            pred_states=all_preds[wn],
            device=device,
            min_errors=min_errors,
        )
    for wn in windows:
        plot_ablation_activity_heatmaps(
            trajs=all_trajs[wn],
            timing=all_timings[wn],
            save_dir=os.path.join(_fig_dir(config), 'ablation_activity'),
            seed=seed, neuron_char=neuron_char,
            ablation_window=wn,
        )

    plot_4_types_heatmaps(
            trajs=all_trajs,
            save_dir=os.path.join(_fig_dir(config), 'intact_activity'),
            seed=seed
        )

    return dict(pca=pca, flows=flows,
                flows_input_cond=flows_input_cond,
                centers=centers,
                pc_range=pc_range,
                panel_summary=panel_summary)


# ═══════════════════════════════════════════════════════════════════
#  14. Top-level entry point
# ═══════════════════════════════════════════════════════════════════

def generate_all_dynamics_figures(
    config: ExperimentConfig,
    models: Dict[int, ContinuousTimeRNN],
    neuron_chars: Dict[int, NeuronCharacterization],
    data_loader,
    seeds: Optional[List[int]] = None,
    windows: Optional[List[str]] = None,
    grid_res: int = GRID_RES_DEFAULT,
    force_rerun: bool = False,
    min_errors: int = 1,
):
    print("\n" + "=" * 60)
    print("  Autonomous + Input-Conditioned Dynamics Analysis")
    print(f"  Adaptive PC3+ grid reconstruction (trajectory-aware)")
    print(f"  q-landscape + speed-proportional quiver | "
          f"{ERROR_DISPLAY}-only ablated trajectories")
    print(f"  5-row × 5-col input-conditioned comparison figure")
    print(f"  min_errors = {min_errors}")
    print("=" * 60)

    if seeds is None:
        seeds = [sorted(models.keys())[0]]
    if windows is None:
        windows = list(DYNAMICS_WINDOWS.keys())

    results = {}
    missing_seeds: List[int] = []
    seed_panel_report: Dict[int, Dict[str, Tuple[int, int]]] = {}

    for seed in seeds:
        if seed not in models:
            print(f"  seed={seed}: model not found — skipping")
            missing_seeds.append(seed)
            continue
        if seed not in neuron_chars:
            print(f"  seed={seed}: neuron_char not found — skipping")
            missing_seeds.append(seed)
            continue
        print(f"\n{'─' * 50}")
        print(f"  Seed {seed}")
        print(f"{'─' * 50}")

        result = analyse_single_seed(
            seed=seed,
            model=models[seed],
            data_loader=data_loader,
            neuron_char=neuron_chars[seed],
            config=config,
            device=config.device,
            windows=windows,
            grid_res=grid_res,
            force_rerun=force_rerun,
            min_errors=min_errors,
        )
        results[seed] = result
        ps = result.get('panel_summary', {})
        for wn, (plotted, skipped) in ps.items():
            seed_panel_report.setdefault(seed, {})[wn] = (
                len(plotted), len(plotted) + len(skipped))

    if missing_seeds:
        print(f"  Missing:  {missing_seeds}")
    partial = [s for s, wr in seed_panel_report.items()
               if any(n < tot for (n, tot) in wr.values())]
    if partial:
        print(f"\n  Partial coverage: {partial}")
    else:
        print(f"\n  ✓ All seeds: full coverage")
    print(f"{'═' * 68}")
    return results


# ═══════════════════════════════════════════════════════════════════
#  15. Standalone CLI
# ═══════════════════════════════════════════════════════════════════

def _load_neuron_char_from_cache(config, seed):
    search_dirs = [
        os.path.join(config.paths.data_dir, 'analysis'),
        os.path.join(config.paths.data_dir, 'analysis', 'neurons'),
        config.paths.data_dir,
    ]
    patterns = [
        f'seed{seed}_neuron_char.pkl',
        f'seed{seed}_neuron_characterization.pkl',
        f'seed{seed}_analysis_result.pkl',
        f'individual_seed{seed}.pkl',
    ]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for pat in patterns:
            path = os.path.join(d, pat)
            if not os.path.exists(path):
                continue
            obj = _load(path)
            if obj is None:
                continue
            if isinstance(obj, NeuronCharacterization):
                return obj, None, None
            nc = getattr(obj, 'neuron_characterization', None)
            if nc is not None:
                return nc
            if isinstance(obj, dict):
                nc = obj.get('neuron_characterization',
                             obj.get('neuron_char'))
                if nc is not None:
                    return nc
    results_path = os.path.join(config.paths.data_dir,
                                'experiment_results.pkl')
    if os.path.exists(results_path):
        obj = _load(results_path)
        if obj is not None and isinstance(obj, dict):
            nc_dict = obj.get('neuron_characterizations', {})
            nc = nc_dict.get(seed)
            if nc is not None and isinstance(nc, NeuronCharacterization):
                return nc
    return None, None, None


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Autonomous + input-conditioned flow-field analysis')
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--all_seeds', action='store_true')
    parser.add_argument('--windows', type=str, nargs='+',
                        default=['init_delay'],
                        choices=['init_delay', 'full_trial'])
    parser.add_argument('--grid_res', type=int, default=GRID_RES_DEFAULT)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--force_rerun', action='store_true')
    args = parser.parse_args()

    from config import get_default_config, print_config
    config = get_default_config(gpu_id=args.gpu)
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.hidden_dim:
        config.model.hidden_dim = args.hidden_dim
    print_config(config)

    from train import load_trained_models
    print("\nLoading trained models ...")
    models = load_trained_models(config)
    if not models:
        print("ERROR: No trained models found.")
        return

    if args.all_seeds:
        seeds = sorted(models.keys())
    elif args.seeds is not None:
        seeds = [s for s in args.seeds if s in models]
    else:
        seeds = None

    from task import load_datasets, create_dataloaders
    bundle = load_datasets(config.paths)
    loaders = create_dataloaders(
        bundle, batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        pin_memory=config.train.pin_memory)

    store = None
    try:
        from analysis_store import AnalysisResultStore
        store = AnalysisResultStore(config)
    except ImportError:
        pass
    if store is None:
        try:
            from analysis_core import ResultStore
            store = ResultStore(config)
        except (ImportError, AttributeError):
            pass

    neuron_chars: Dict[int, NeuronCharacterization] = {}
    target_seeds = seeds if seeds is not None else sorted(models.keys())

    for seed in target_seeds:
        loaded = False
        if store is not None:
            try:
                res = store.load_individual(seed)
                if (res is not None
                        and hasattr(res, 'neuron_characterization')
                        and res.neuron_characterization is not None):
                    neuron_chars[seed] = res.neuron_characterization
                    loaded = True
            except Exception as e:
                print(f"  seed={seed}: store failed: {e}")
        if not loaded:
            nc  = _load_neuron_char_from_cache(config, seed)
            if nc is not None:
                neuron_chars[seed] = nc
                loaded = True
        if loaded:
            print(f"  seed={seed}: loaded")
        else:
            print(f"  Warning: seed={seed} no neuron char")

    if not neuron_chars:
        print("ERROR: No neuron characterisations found.")
        return

    generate_all_dynamics_figures(
        config=config, models=models,
        neuron_chars=neuron_chars,
        data_loader=loaders['ablation'],
        seeds=seeds, windows=args.windows,
        grid_res=args.grid_res,
        force_rerun=args.force_rerun
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
