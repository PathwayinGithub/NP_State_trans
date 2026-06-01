import os
import pickle
import argparse
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from sklearn.cross_decomposition import PLSRegression
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from tqdm import tqdm

from model import ContinuousTimeRNN
from config import get_default_config, print_config, ExperimentConfig
from analysis_core import (
    TRIAL_TYPES, TRIAL_COLORS, TRIAL_SHORT,
    EXPECTED_STATE_MAP,
    resolve_resp_start,
    classify_trial_state,
    set_publication_style, save_figure,
)
from analysis_neurons import (
    NeuronCharacterization,
    build_neuron_ablation_conditions,
)


# ═══════════════════════════════════════════════════════════════════
#  Constants (migrated from vis_dynamics_log_velocity + local)
# ═══════════════════════════════════════════════════════════════════

ERROR_DISPLAY: str = 'error'
N_INDIVIDUAL_SHOW  = 20
N_TRAJ_PER_TYPE    = 400

EXTEND_N_STEPS     = 50
EXTEND_COLOR       = '#00E676'
EXTEND_LW_IND     = 0.55
EXTEND_LW_MEAN    = 2.8
EXTEND_ALPHA_IND  = 0.18
EXTEND_ALPHA_MEAN = 0.85

_PHASE_PROPS = OrderedDict([
    ('init',     {'ls': '-',  'lw_f': 0.40, 'a_f': 0.50, 'arrow_scale': 5,  'arrow_style': '-|>'}),
    ('delay',    {'ls': '--', 'lw_f': 0.68, 'a_f': 0.75, 'arrow_scale': 6,  'arrow_style': '-|>'}),
    ('response', {'ls': '-',  'lw_f': 1.00, 'a_f': 1.00, 'arrow_scale': 11, 'arrow_style': 'simple'}),
])

FONT = {
    'suptitle': 16, 'title': 14, 'label': 12,
    'tick': 11, 'legend': 10, 'annot': 10, 'bar_text': 10,
}

# PLSR-specific
JOINT_FIT          = True
TEMPLATE_SIGMA     = 3.0
MAX_PLS_COMPONENTS = 20
GRID_RES           = 60
N_QUIVER           = 22
QUIVER_N_STEPS     = 5
PADDING            = 0.15

CMAP_HM    = 'RdBu_r'
CMAP_TMPL  = mcolors.LinearSegmentedColormap.from_list(
    'tmpl', ['white', '#D95F4B'], N=256)
CMAP_LAND  = mcolors.LinearSegmentedColormap.from_list(
    'flow', ['#5B9FD4', '#F4EFE9', '#D95F4B'], N=512)

DROPOUT_ORANGE = '#E8A33D'


# ═══════════════════════════════════════════════════════════════════
#  IO helpers
# ═══════════════════════════════════════════════════════════════════

def _cache_dir(cfg):
    d = os.path.join(cfg.paths.data_dir, 'analysis', 'plsr_dynamics')
    os.makedirs(d, exist_ok=True)
    return d


def _fig_dir(cfg):
    d = os.path.join(cfg.paths.figure_dir, 'plsr_dynamics')
    os.makedirs(d, exist_ok=True)
    return d


def _save(o, p):
    with open(p, 'wb') as f:
        pickle.dump(o, f)
    print(f"    cached → {os.path.basename(p)}")


def _load(p):
    if os.path.exists(p):
        with open(p, 'rb') as f:
            return pickle.load(f)
    return None


# ═══════════════════════════════════════════════════════════════════
#  Direction-arrow helpers (migrated)
# ═══════════════════════════════════════════════════════════════════

def _add_traj_arrow_mid(ax, mean_pc, color, alpha, lw,
                        head_scale: float = 8, n_trim: int = 3,
                        arrowstyle: str = '-|>'):  
    T = len(mean_pc)
    if T < n_trim + 2:
        return 0
    i0 = T - 1 - n_trim
    i1 = T - 1
    ax.annotate(
        '', xy=(mean_pc[i1, 0], mean_pc[i1, 1]),
        xytext=(mean_pc[i0, 0], mean_pc[i0, 1]),
        arrowprops=dict(arrowstyle=arrowstyle, 
                        color=color, lw=lw,
                        mutation_scale=head_scale,
                        shrinkA=0, shrinkB=0),
        alpha=alpha, zorder=6)
    return n_trim


def _add_segment_end_arrow(ax, seg_pc, color, alpha, lw,
                           head_scale: float = 6, n_span: int = 2,
                           arrowstyle: str = '-|>'):  
    T = len(seg_pc)
    if T < n_span + 1:
        return
    i0 = T - 1 - n_span
    i1 = T - 1
    ax.annotate(
        '', xy=(seg_pc[i1, 0], seg_pc[i1, 1]),
        xytext=(seg_pc[i0, 0], seg_pc[i0, 1]),
        arrowprops=dict(arrowstyle=arrowstyle,    
                        color=color, lw=lw * 0.8,
                        mutation_scale=head_scale,
                        shrinkA=0, shrinkB=0),
        alpha=alpha, zorder=6)


# ═══════════════════════════════════════════════════════════════════
#  Phased trajectory drawer (migrated)
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
        ('delay',    max(0, init_s),             min(init_s + delay_s + 1, T)),
        ('response', max(0, init_s + delay_s),   T),
    ]

    _PHASE_LINE_COLOR = {
        'init':     'black',
        'delay':    'black',
        'response': color,
    }

    for phase_name, s_start, s_end in segments:
        seg = mean_pc[s_start:s_end]
        if len(seg) < 2:
            continue
        props    = _PHASE_PROPS[phase_name]
        lw       = base_lw    * props['lw_f']
        alpha    = base_alpha * props['a_f']
        pc       = _PHASE_LINE_COLOR[phase_name] 
        is_last  = (phase_name == 'response')
        n_trim   = 3 if (is_last and len(seg) > 5) else 0
        plot_seg = seg[:len(seg) - n_trim] if n_trim > 0 else seg
        
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
#  Legend builder (migrated)
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
#  Speed-proportional quiver helper (migrated)
# ═══════════════════════════════════════════════════════════════════

def _draw_speed_quiver(ax, Xq, Yq, U_raw, V_raw):
    finite = np.isfinite(U_raw) & np.isfinite(V_raw)
    spd = np.sqrt(
        np.where(finite, U_raw, 0.0) ** 2 +
        np.where(finite, V_raw, 0.0) ** 2
    )

    valid_spd = spd[finite & (spd > 1e-10)]
    ref = float(np.percentile(valid_spd, 85)) if valid_spd.size > 0 else 1.0

    log_norm = np.log1p(1.0)
    mag = np.log1p(spd / (ref + 1e-12)) / log_norm
    mag = np.clip(mag, 0.0, 2.5)
    mag = np.where(mag < 0.02, 0.0, mag)

    spd_safe = np.where(spd > 1e-12, spd, 1.0)
    Ux = (U_raw / spd_safe) * mag
    Vy = (V_raw / spd_safe) * mag
    Ux = np.where(finite, Ux, np.nan)
    Vy = np.where(finite, Vy, np.nan)

    pc1_min, pc1_max = Xq.min(), Xq.max()
    pc2_min, pc2_max = Yq.min(), Yq.max()
    n_cols = Xq.shape[1] - 1
    n_rows = Yq.shape[0] - 1
    grid_step = max(
        (pc1_max - pc1_min) / max(n_cols, 1),
        (pc2_max - pc2_min) / max(n_rows, 1),
    )
    scale_xy = 1.0 / (grid_step + 1e-12)

    ax.quiver(
        Xq, Yq, Ux, Vy,
        color='#3b3b3b', alpha=0.80,
        width=0.0028,
        headwidth=3.8,
        headlength=4.5,
        headaxislength=4.0,
        minshaft=1.2,
        minlength=0.0,
        scale=scale_xy,
        scale_units='xy',
        angles='xy',
        pivot='tail',
        zorder=2,
    )


# ═══════════════════════════════════════════════════════════════════
#  Autonomous step helper (migrated)
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
#  Filter error trials (migrated)
# ═══════════════════════════════════════════════════════════════════

def filter_error_trials(
    trajectory_list: List[np.ndarray],
    predicted_states: List[str],
    trial_type: str,
) -> Tuple[List[np.ndarray], int, int]:
    expected = EXPECTED_STATE_MAP[trial_type]
    assert len(trajectory_list) == len(predicted_states)
    filtered = [arr for arr, p in zip(trajectory_list, predicted_states)
                if p != expected]
    return filtered, len(filtered), len(trajectory_list)


# ═══════════════════════════════════════════════════════════════════
#  Trajectory collection (migrated)
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
        n_err, n_tot = _summarise_error_counts(pred_states, cn)
        if n_tot > 0:
            print(f"      {cn}: {n_err}/{n_tot} {ERROR_DISPLAY} "
                  f"({100 * n_err / n_tot:.1f}%)")

    return trajs, timing, pred_states


def _summarise_error_counts(
    pred_states: Dict[str, Dict[str, List[str]]],
    condition: str,
) -> Tuple[int, int]:
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
#  Extend trajectories (migrated)
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


# ═══════════════════════════════════════════════════════════════════
#  Load neuron char from cache (migrated)
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
                return obj
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
    return None


# ═══════════════════════════════════════════════════════════════════
#  Data preparation
# ═══════════════════════════════════════════════════════════════════

def _trial_mean(arrs: List[np.ndarray],
                skip: int = 0) -> Optional[np.ndarray]:
    if not arrs:
        return None
    L = min(a.shape[0] for a in arrs)
    if L - skip < 2:
        return None
    return np.stack([a[skip:L] for a in arrs]).mean(0)


def build_sequential_template(
    X: np.ndarray, sigma: float = TEMPLATE_SIGMA,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T, N = X.shape
    peaks = np.argmax(X, axis=0)
    sort_order = np.argsort(peaks)
    t_grid = np.arange(T)
    Y = np.exp(-0.5 * ((t_grid[:, None] - peaks[None, :]) / sigma) ** 2)
    return Y.astype(np.float32), peaks, sort_order


def zscore_columns(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + eps)


# ═══════════════════════════════════════════════════════════════════
#  PLSR helpers
# ═══════════════════════════════════════════════════════════════════

def _pls_x_stats(pls: PLSRegression) -> Tuple[np.ndarray, np.ndarray]:
    x_mean = (getattr(pls, '_x_mean', None)
              if getattr(pls, '_x_mean', None) is not None
              else getattr(pls, 'x_mean_', None))
    x_std  = (getattr(pls, '_x_std',  None)
              if getattr(pls, '_x_std',  None) is not None
              else getattr(pls, 'x_std_',  None))
    if x_mean is None:
        x_mean = np.zeros(pls.x_loadings_.shape[0], dtype=np.float32)
    if x_std is None:
        x_std = np.ones_like(x_mean)
    return np.asarray(x_mean), np.asarray(x_std)


def pls_inverse(scores: np.ndarray, pls: PLSRegression) -> np.ndarray:
    x_mean, x_std = _pls_x_stats(pls)
    return scores @ pls.x_loadings_.T * x_std + x_mean


def fit_plsr(X: np.ndarray, Y: np.ndarray, k: int) -> PLSRegression:
    return PLSRegression(n_components=k, scale=True,
                         max_iter=1000, tol=1e-7).fit(X, Y)


def _collect_autonomous_points(
    trajs: Dict, pls: PLSRegression, timing: Dict,
    use_intact_only: bool = True,
    margin_start: int = 1,
    margin_end:   int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    init_s = int(timing['init_steps'])
    t_start = init_s + margin_start

    pts_low, pts_high = [], []
    sources = ['intact'] if use_intact_only else list(trajs.keys())

    n_trials = 0
    for src in sources:
        cd = trajs.get(src, {})
        for tt, arrs in cd.items():
            for arr in arrs:
                T  = arr.shape[0]
                ts = max(0, t_start)
                te = max(ts, T - margin_end)
                if te - ts < 2:
                    continue
                seg = arr[ts:te]
                pts_high.append(seg)
                pts_low.append(pls.transform(seg))
                n_trials += 1

    if not pts_high:
        return (np.empty((0, 2)),
                np.empty((0, pls.x_loadings_.shape[0])))

    pts_low  = np.vstack(pts_low)
    pts_high = np.vstack(pts_high)
    print(f"      autonomous pool: {pts_high.shape[0]} samples "
          f"from {n_trials} trials (delay+response, x=0)")
    return pts_low, pts_high


# ═══════════════════════════════════════════════════════════════════
#  Dimension dropout test
# ═══════════════════════════════════════════════════════════════════

def dimension_dropout_test(
    X: np.ndarray, Y: np.ndarray,
    max_components: int = MAX_PLS_COMPONENTS,
    desc: str = '  PLS dropout',
) -> Dict[str, np.ndarray]:
    K = min(max_components, min(X.shape) - 1)
    ks = np.arange(1, K + 1)
    diss     = np.zeros(K)
    var_inc  = np.zeros(K)
    var_cum  = np.zeros(K)

    total_var = X.var(axis=0).sum()
    prev_cum  = 0.0
    for i, k in enumerate(tqdm(ks, desc=desc, leave=False)):
        pls_k = fit_plsr(X, Y, k)
        scores = pls_k.transform(X)
        X_hat  = pls_inverse(scores, pls_k)

        r = np.corrcoef(X.ravel(), X_hat.ravel())[0, 1]
        diss[i] = 1.0 - r

        resid_var = (X - X_hat).var(axis=0).sum()
        cum = max(0.0, 1.0 - resid_var / total_var) * 100
        var_cum[i] = cum
        var_inc[i] = cum - prev_cum
        prev_cum = cum

    return dict(components=ks, dissimilarity=diss,
                variance_explained_inc=var_inc,
                variance_explained_cum=var_cum)


def select_k_components(test: Dict[str, np.ndarray],
                        diss_drop_frac: float = 0.85,
                        min_k: int = 2) -> int:
    diss = test['dissimilarity']
    if len(diss) < 2:
        return min_k
    drop_total = diss[0] - diss[-1]
    if drop_total <= 1e-9:
        return min_k
    target = diss[0] - diss_drop_frac * drop_total
    for k_idx, d in enumerate(diss):
        if d <= target:
            return max(min_k, k_idx + 1)
    return min(len(diss), min_k + 3)


# ═══════════════════════════════════════════════════════════════════
#  Trajectory projection / flow field in PLS space
# ═══════════════════════════════════════════════════════════════════

def project_trajs(trajs: Dict, pls: PLSRegression) -> Dict:
    return {
        cn: {tt: [pls.transform(arr) for arr in arrs]
             for tt, arrs in cd.items()}
        for cn, cd in trajs.items()
    }


def _get_pls_range(proj_trajs: Dict) -> Tuple[float, float, float, float]:
    pts = []
    for cd in proj_trajs.values():
        for arrs in cd.values():
            for arr in arrs:
                pts.append(arr[:, :2])
    pts = np.vstack(pts)
    return pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()


def _reconstruct_grid_pls_adaptive(
    grid_2d: np.ndarray, pls: PLSRegression,
    traj_low: np.ndarray, traj_high: np.ndarray,
    k_neighbors: int = 50, sigma_weight: float = 1.5,
) -> np.ndarray:
    K = pls.n_components
    n_grid = grid_2d.shape[0]
    M = traj_low.shape[0]

    if M < max(k_neighbors, 4):
        scores = np.zeros((n_grid, K), dtype=np.float32)
        scores[:, 0] = grid_2d[:, 0]
        scores[:, 1] = grid_2d[:, 1]
        return pls_inverse(scores, pls).astype(np.float32)

    k_use = min(k_neighbors, M)
    tree = cKDTree(traj_low[:, :2])
    dists, idxs = tree.query(grid_2d, k=k_use)
    if k_use == 1:
        dists = dists[:, None]; idxs = idxs[:, None]

    w = np.exp(-0.5 * (dists / sigma_weight) ** 2)
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum = np.where(w_sum > 1e-12, w_sum, 1.0)
    w_norm = (w / w_sum).astype(np.float32)

    h_grid = np.einsum('ij,ijk->ik',
                       w_norm,
                       traj_high[idxs]).astype(np.float32)
    return h_grid


def _grid_pls(pls_range, n: int, padding: float, K: int,
              proj_trajs: Optional[Dict] = None):
    pmin1, pmax1, pmin2, pmax2 = pls_range
    pad1 = (pmax1 - pmin1) * padding
    pad2 = (pmax2 - pmin2) * padding
    xi = np.linspace(pmin1 - pad1, pmax1 + pad1, n)
    yi = np.linspace(pmin2 - pad2, pmax2 + pad2, n)
    Xq, Yq = np.meshgrid(xi, yi)
    n_grid = Xq.size
    grid = np.zeros((n_grid, K), dtype=np.float32)
    grid[:, 0] = Xq.ravel()
    grid[:, 1] = Yq.ravel()
    if K > 2 and proj_trajs is not None:
        pts = []
        for cd in proj_trajs.values():
            for arrs in cd.values():
                for arr in arrs:
                    pts.append(arr)
        if pts:
            z_const = np.median(np.vstack(pts)[:, 2:], axis=0)
            for k in range(2, K):
                grid[:, k] = z_const[k - 2]
    return Xq, Yq, grid


def compute_landscape_pls(model, pls, pls_range, device,
                          trajs, proj_trajs, timing,
                          grid_res=GRID_RES, padding=PADDING) -> Dict:
    K = pls.n_components
    Xi, Yi, grid = _grid_pls(pls_range, grid_res, padding, K, proj_trajs)

    pts_low, pts_high = _collect_autonomous_points(
        trajs, pls, timing,
        use_intact_only=True,
        margin_start=1, margin_end=0,
    )
    print(f"    landscape kNN pool: {pts_high.shape[0]} autonomous points")

    h_grid = _reconstruct_grid_pls_adaptive(
        grid_2d=grid[:, :2], pls=pls,
        traj_low=pts_low, traj_high=pts_high,
        k_neighbors=50, sigma_weight=1.5,
    )

    model.eval()
    h0 = torch.tensor(h_grid, dtype=torch.float32, device=device)
    with torch.no_grad():
        h1 = _autonomous_step_maybe_ablated(model, h0, None, None)
        dh = (h1 - h0).cpu().numpy()
    speed     = np.sqrt((dh ** 2).sum(axis=1))
    log_speed = np.log10(speed + 1e-12).reshape(grid_res, grid_res)
    Sc        = gaussian_filter(log_speed, sigma=1.0)
    return dict(Xi=Xi, Yi=Yi, Sc=Sc)


def compute_quiver_pls(model, pls, pls_range, device,
                       trajs, proj_trajs, timing,
                       n_quiver=N_QUIVER, padding=PADDING,
                       n_steps=QUIVER_N_STEPS) -> Dict:
    K = pls.n_components
    Xq, Yq, grid = _grid_pls(pls_range, n_quiver, padding, K, proj_trajs)

    pts_low, pts_high = _collect_autonomous_points(
        trajs, pls, timing,
        use_intact_only=True,
        margin_start=1, margin_end=0,
    )
    print(f"    quiver kNN pool: {pts_high.shape[0]} autonomous points")

    h_grid = _reconstruct_grid_pls_adaptive(
        grid_2d=grid[:, :2], pls=pls,
        traj_low=pts_low, traj_high=pts_high,
        k_neighbors=20, sigma_weight=0.8,
    )

    model.eval()
    h = torch.tensor(h_grid, dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(n_steps):
            h = _autonomous_step_maybe_ablated(model, h, None, None)
    end_pls = pls.transform(h.cpu().numpy())
    dU = (end_pls[:, 0] - grid[:, 0]).reshape(n_quiver, n_quiver)
    dV = (end_pls[:, 1] - grid[:, 1]).reshape(n_quiver, n_quiver)
    return dict(Xq=Xq, Yq=Yq, U=dU, V=dV)


# ═══════════════════════════════════════════════════════════════════
#  Attractor centres in PLS space
# ═══════════════════════════════════════════════════════════════════

def _compute_pls_centers(trajs_intact: Dict, pls: PLSRegression) -> Dict:
    nrem_ep, wake_ep = [], []
    for arr in trajs_intact.get('nrem_only', []):
        nrem_ep.append(arr[-1]); nrem_ep.append(arr[0])
    for arr in trajs_intact.get('wake_only', []):
        wake_ep.append(arr[-1]); wake_ep.append(arr[0])
    for arr in trajs_intact.get('wake_to_nrem', []):
        nrem_ep.append(arr[-1]); wake_ep.append(arr[0])
    for arr in trajs_intact.get('nrem_to_wake', []):
        wake_ep.append(arr[-1]); nrem_ep.append(arr[0])

    centers = {}
    if nrem_ep:
        pts = pls.transform(np.array(nrem_ep))[:, :2]
        centers['NREM'] = (float(pts[:, 0].mean()),
                           float(pts[:, 1].mean()))
    if wake_ep:
        pts = pls.transform(np.array(wake_ep))[:, :2]
        centers['Wake'] = (float(pts[:, 0].mean()),
                           float(pts[:, 1].mean()))
    return centers


# ═══════════════════════════════════════════════════════════════════
#  Plotting — heatmap (top-left → bottom-right)
# ═══════════════════════════════════════════════════════════════════

def _plot_heatmap(ax, data, title, sort_order, cmap=CMAP_HM,
                  vmin=-2.5, vmax=2.5, show_ylabel=True,
                  show_xlabel=True, show_title=True,
                  ylabel='Neurons'):
    M = data.T[sort_order]
    N, T = M.shape
    im = ax.imshow(M, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest', origin='upper')

    tick_pos = [0, N // 4, N // 2, 3 * N // 4, N - 1]
    tick_pos = sorted(set(tick_pos)) 
    tick_lab = [str(sort_order[i] + 1) for i in tick_pos]
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_lab, fontsize=7)
    # ────────────────────────────────────────────────────────────

    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if show_xlabel:
        ax.set_xlabel('Time (bins)', fontsize=9)
    if show_title:
        ax.set_title(title, fontsize=11, fontweight='bold')
    return im

# ═══════════════════════════════════════════════════════════════════
#  Plotting — single dropout-test panel
# ═══════════════════════════════════════════════════════════════════

def _plot_dropout_single(
    ax, test, k_chosen,
    title='Dimension dropout test',
    show_xlabel=True,
    show_arrow=True,
    show_ylabels=True,
):
    ks      = test['components']
    diss    = test['dissimilarity']
    var_inc = test['variance_explained_inc']
    K_max   = int(ks.max())

    ax.axvspan(0, k_chosen + 0.5,
               color='#FCE7B0', alpha=0.55, zorder=1)

    ax.plot(ks, diss, color='black', lw=1.8,
            marker='o', ms=4.5, zorder=3,
            markerfacecolor='black', markeredgecolor='black')
    if show_ylabels:
        ax.set_ylabel('Dissimilarity from\ninput data',
                      fontsize=9, color='black')
    ax.tick_params(axis='y', labelcolor='black', labelsize=8)
    ax.set_xlim(0, K_max + 0.5)
    ax.set_ylim(0, max(diss.max() * 1.10, 0.05))

    if show_xlabel:
        ax.set_xlabel('PLS dimensions', fontsize=10)
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis='x', labelsize=8)

    ax2 = ax.twinx()
    ax2.plot(ks, var_inc, color=DROPOUT_ORANGE, lw=1.8,
             marker='o', ms=4.5, zorder=3,
             markerfacecolor=DROPOUT_ORANGE,
             markeredgecolor=DROPOUT_ORANGE)
    if show_ylabels:
        ax2.set_ylabel('Variance explained (%)',
                       fontsize=9, color=DROPOUT_ORANGE)
    ax2.tick_params(axis='y', labelcolor=DROPOUT_ORANGE, labelsize=8)
    ax2.set_ylim(0, max(var_inc.max() * 1.15, 1.0))

    if show_arrow:
        diss_top = diss.max()
        x_text   = k_chosen + 2.5
        y_text   = diss_top * 0.70
        x_arrow  = k_chosen * 0.55
        y_arrow  = diss_top * 0.55
        ax.annotate(
            'Used for\nlater analysis',
            xy=(x_arrow, y_arrow),
            xytext=(x_text, y_text),
            fontsize=8.5, ha='left', va='center',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
            zorder=5)

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')


# ═══════════════════════════════════════════════════════════════════
#  Plotting — PLS trajectory panel (standard, no extension)
# ═══════════════════════════════════════════════════════════════════

def _plot_pls_2d(
    ax, flow, proj_intact, proj_error,
    tt, timing, centers,
    show_legend=True,
    n_individual=N_INDIVIDUAL_SHOW,
    contour_levels=None,
):
    init_s  = timing['init_steps']
    delay_s = timing['delay_steps']
    Xi, Yi, Sc = flow['Xi'], flow['Yi'], flow['Sc']

    if contour_levels is not None:
        cf = ax.contourf(Xi, Yi, Sc, levels=contour_levels,
                         cmap=CMAP_LAND, extend='neither')
        ax.contour(Xi, Yi, Sc, levels=contour_levels[::2],
                   colors='white', alpha=0.25, linewidths=0.4)
    else:
        cf = ax.contourf(Xi, Yi, Sc, levels=25, cmap=CMAP_LAND)
        ax.contour(Xi, Yi, Sc, levels=12, colors='white',
                   alpha=0.25, linewidths=0.4)

    if 'Xq' in flow and flow.get('U') is not None:
        _draw_speed_quiver(ax, flow['Xq'], flow['Yq'],
                           flow['U'], flow['V'])

    color = TRIAL_COLORS[tt]
    ablated_plotted: List[str] = []

    if proj_intact:
        L_min = min(a.shape[0] for a in proj_intact)
        if L_min >= 2:
            mean_i = np.stack([a[:L_min, :2]
                               for a in proj_intact]).mean(0)
            _draw_phased_trajectory(ax, mean_i, color,
                                    init_s, delay_s,
                                    base_lw=3.0, base_alpha=0.95)

    if proj_error:
        n_show = min(len(proj_error), n_individual)
        for arr in proj_error[:n_show]:
            ax.plot(arr[:, 0], arr[:, 1],
                    color=color, lw=1.0, alpha=0.45, zorder=2.5,
                    solid_capstyle='round')
            ax.scatter(arr[-1, 0], arr[-1, 1],
                    c=color, s=22, alpha=0.65,
                    edgecolors='none', zorder=2.8)
        L_min = min(a.shape[0] for a in proj_error)
        if L_min >= 2:
            mean_e = np.stack([a[:L_min, :2]
                               for a in proj_error]).mean(0)
            _draw_phased_trajectory(ax, mean_e, color,
                                    init_s, delay_s,
                                    base_lw=2.4, base_alpha=0.60)
            ablated_plotted.append(tt)

    _ac = {'NREM': '#27ae60', 'Wake': '#f39c12'}
    for name, (p1c, p2c) in (centers or {}).items():
        col = _ac.get(name, '#9b59b6')
        ax.scatter(p1c, p2c, c=col, s=130, marker='o',
                   edgecolors='k', linewidths=1.2, zorder=7)
        ax.text(p1c, p2c + 0.3, name, color='black',
                fontsize=7, fontweight='bold',
                ha='center', va='bottom', zorder=8)

    ax.set_xlabel('PLS-1', fontsize=9)
    ax.set_ylabel('PLS-2', fontsize=9)
    ax.set_title(
        f'{TRIAL_SHORT[tt]}: intact vs {ERROR_DISPLAY} in PLS space',
        fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=7)

    if show_legend:
        handles = _build_legend_handles(
            show_tt=[tt], show_ablated=True,
            ablated_tt=ablated_plotted)
        leg = ax.legend(
            handles=handles, loc='upper right',
            fontsize=6, ncol=2,
            framealpha=0.92, edgecolor='gray', fancybox=True,
            borderpad=0.4, handlelength=1.2, labelspacing=0.25,
            columnspacing=0.8, handletextpad=0.4)
        leg.set_zorder(10)
    return cf


# ═══════════════════════════════════════════════════════════════════
#  Plotting — PLS trajectory panel WITH extended tail
# ═══════════════════════════════════════════════════════════════════

def _plot_pls_2d_extended(
    ax, flow,
    proj_intact: List[np.ndarray],
    proj_ext: List[np.ndarray],
    tt: str,
    timing: Dict,
    centers: Dict,
    original_total_len: int,
    n_extra_steps: int = EXTEND_N_STEPS,
    show_legend: bool = True,
    n_individual: int = N_INDIVIDUAL_SHOW,
    contour_levels: Optional[np.ndarray] = None,
):
    """
    Like _plot_pls_2d but draws extended (autonomous) tails in green
    after the original trial length, similar to _plot_2d_extended.
    """
    init_s  = timing['init_steps']
    delay_s = timing['delay_steps']
    Xi, Yi, Sc = flow['Xi'], flow['Yi'], flow['Sc']

    # ── background contour ──
    if contour_levels is not None:
        cf = ax.contourf(Xi, Yi, Sc, levels=contour_levels,
                         cmap=CMAP_LAND, extend='neither')
        ax.contour(Xi, Yi, Sc, levels=contour_levels[::2],
                   colors='white', alpha=0.25, linewidths=0.4)
    else:
        cf = ax.contourf(Xi, Yi, Sc, levels=25, cmap=CMAP_LAND)
        ax.contour(Xi, Yi, Sc, levels=12, colors='white',
                   alpha=0.25, linewidths=0.4)

    # ── quiver ──
    if 'Xq' in flow and flow.get('U') is not None:
        _draw_speed_quiver(ax, flow['Xq'], flow['Yq'],
                           flow['U'], flow['V'])

    color = TRIAL_COLORS[tt]
    T_orig = original_total_len
    ablated_plotted: List[str] = []

    # ── intact mean trajectory ──
    if proj_intact:
        L_min = min(a.shape[0] for a in proj_intact)
        if L_min >= 2:
            mean_i = np.stack([a[:L_min, :2]
                               for a in proj_intact]).mean(0)
            _draw_phased_trajectory(ax, mean_i, color,
                                    init_s, delay_s,
                                    base_lw=3.0, base_alpha=0.95)

    # ── individual extended error trajectories ──
    if proj_ext:
        n_show = min(len(proj_ext), n_individual)
        for arr in proj_ext[:n_show]:
            pc_i = arr[:, :2]
            t_cut = min(T_orig, len(pc_i))

            # original portion
            ax.plot(pc_i[:t_cut, 0], pc_i[:t_cut, 1],
                color=color, lw=1.0, alpha=0.45, zorder=2.5,
                solid_capstyle='round')
            
            # extended portion (green)
            if len(pc_i) > t_cut:
                ax.plot(pc_i[t_cut - 1:, 0], pc_i[t_cut - 1:, 1],
                        color=EXTEND_COLOR, lw=EXTEND_LW_IND,
                        alpha=EXTEND_ALPHA_IND, zorder=2.6,
                        solid_capstyle='round')
                ax.scatter(pc_i[-1, 0], pc_i[-1, 1],
                           c=EXTEND_COLOR, s=12, alpha=0.35,
                           edgecolors='none', zorder=2.8)

        # ── mean of error trajectories: original portion (phased) ──
        L_min_orig = min(min(a.shape[0], T_orig) for a in proj_ext)
        if L_min_orig >= 2:
            mean_orig = np.stack(
                [a[:L_min_orig, :2] for a in proj_ext]).mean(0)
            _draw_phased_trajectory(ax, mean_orig, color,
                                    init_s, delay_s,
                                    base_lw=2.4, base_alpha=0.60)
            ablated_plotted.append(tt)

        # ── mean of error trajectories: extended portion (green) ──
        ext_segs = []
        for a in proj_ext:
            if a.shape[0] > T_orig:
                ext_segs.append(a[max(0, T_orig - 1):, :2])
        if ext_segs:
            L_ext_min = min(s.shape[0] for s in ext_segs)
            if L_ext_min >= 3:
                mean_ext = np.stack(
                    [s[:L_ext_min] for s in ext_segs]).mean(0)
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
    for name, (p1c, p2c) in (centers or {}).items():
        col = _ac.get(name, '#9b59b6')
        ax.scatter(p1c, p2c, c=col, s=130, marker='o',
                   edgecolors='k', linewidths=1.2, zorder=7)
        ax.text(p1c, p2c + 0.3, name, color='black',
                fontsize=7, fontweight='bold',
                ha='center', va='bottom', zorder=8)

    ax.set_xlabel('PLS-1', fontsize=9)
    ax.set_ylabel('PLS-2', fontsize=9)
    ax.set_title(
        f'{TRIAL_SHORT[tt]}: extended error trajectories '
        f'(+{n_extra_steps} steps)',
        fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=7)

    if show_legend:
        handles = _build_legend_handles(
            show_tt=[tt], show_ablated=True,
            ablated_tt=ablated_plotted)
        handles.append(plt.Line2D(
            [], [], color=EXTEND_COLOR, lw=2.5,
            alpha=EXTEND_ALPHA_MEAN,
            label=f'+{n_extra_steps} auton. steps'))
        handles.append(plt.Line2D(
            [], [], color=EXTEND_COLOR, marker='*', markersize=8,
            linestyle='none', label='Extended endpoint'))
        leg = ax.legend(
            handles=handles, loc='upper right',
            fontsize=5.5, ncol=2,
            framealpha=0.92, edgecolor='gray', fancybox=True,
            borderpad=0.4, handlelength=1.2, labelspacing=0.25,
            columnspacing=0.8, handletextpad=0.4)
        leg.set_zorder(10)
    return cf


# ═══════════════════════════════════════════════════════════════════
#  Generate extended-trajectory figure (PLS version)
# ═══════════════════════════════════════════════════════════════════

def generate_pls_extended_figure(
    seed: int,
    config: ExperimentConfig,
    model: ContinuousTimeRNN,
    pls: PLSRegression,
    landscape: Dict,
    quiver: Dict,
    trajs: Dict,
    pred_states: Dict,
    proj_trajs: Dict,
    timing: Dict,
    centers: Dict,
    device: str,
    n_extra_steps: int = EXTEND_N_STEPS,
):
    """
    Generate a 1×2 figure showing extended trajectories in PLS space
    for N→W (ablate_n2w) and W→N (ablate_w2n).
    """
    set_publication_style()

    TT_TO_ABL = {
        'nrem_to_wake': 'ablate_n2w',
        'wake_to_nrem': 'ablate_w2n',
    }

    total_s = timing['total_steps']
    original_total_len = total_s + 1

    # ── extend error trajectories ──
    print(f"    Extending error trajectories by "
          f"{n_extra_steps} autonomous steps (PLS figure) ...")
    ext_proj: Dict[str, List[np.ndarray]] = {}
    proj_intact_for_plot: Dict[str, List[np.ndarray]] = {}

    for tt, abl_key in TT_TO_ABL.items():
        abl_arrs  = trajs.get(abl_key, {}).get(tt, [])
        abl_preds = pred_states.get(abl_key, {}).get(tt, [])

        # intact projections
        proj_intact_for_plot[tt] = proj_trajs.get('intact', {}).get(tt, [])

        if not abl_arrs or not abl_preds:
            ext_proj[tt] = []
            print(f"      {abl_key}/{tt}: no trials available")
            continue

        err_arrs, n_err, n_tot = filter_error_trials(
            abl_arrs, abl_preds, tt)
        print(f"      {abl_key}/{tt}: {n_err}/{n_tot} error trials")

        if not err_arrs:
            ext_proj[tt] = []
            continue

        # extend in hidden space
        ext_arrs = _extend_trajectories_autonomous(
            model, err_arrs,
            n_extra_steps=n_extra_steps,
            device=device)
        print(f"        extended: {len(ext_arrs)} trajs  "
              f"T={original_total_len} → "
              f"{original_total_len + n_extra_steps}")

        # project to PLS
        ext_proj[tt] = [pls.transform(a) for a in ext_arrs]

    # ── check we have something to plot ──
    valid_tts = [tt for tt in TT_TO_ABL if ext_proj.get(tt)]
    if not valid_tts:
        print("    !!! Note: No error trials to extend — skipping extended figure (Not error)")
        return

    # ── figure ──
    n_cols = len(valid_tts)
    fig, axes = plt.subplots(1, n_cols, figsize=(7.0 * n_cols, 6.5))
    if n_cols == 1:
        axes = [axes]

    flow_merged = dict(landscape)
    flow_merged.update(quiver)

    for ci, tt in enumerate(valid_tts):
        ax = axes[ci]
        _plot_pls_2d_extended(
            ax, flow_merged,
            proj_intact=proj_intact_for_plot.get(tt, []),
            proj_ext=ext_proj[tt],
            tt=tt,
            timing=timing,
            centers=centers,
            original_total_len=original_total_len,
            n_extra_steps=n_extra_steps,
            show_legend=True,
        )

    fig.suptitle(
        f'PLS Extended Trajectories (+{n_extra_steps} autonomous steps)  │  '
        f'Seed {seed}',
        fontsize=FONT['suptitle'], fontweight='bold', y=1.01)

    plt.tight_layout()
    save_path = os.path.join(
        _fig_dir(config),
        f'plsr_extended_traj_seed{seed}.png')
    save_figure(fig, save_path)
    plt.close(fig)
    print(f"  ✓ PLS extended trajectory figure: {save_path}")


# ═══════════════════════════════════════════════════════════════════
#  plot_4_types_heatmaps (migrated, with z-score normalization)
# ═══════════════════════════════════════════════════════════════════

def plot_4_types_heatmaps(trajs, save_dir=None, seed=0, timing=None):
    """
    Plot intact neural activity heatmaps for 4 trial types.
    Neurons sorted by peak-activation time to emphasise sequential structure.
    Raw activation values with percentile-based color limits.
    """
    set_publication_style()

    TT_ORDER  = ['nrem_to_wake', 'wake_to_nrem', 'wake_only', 'nrem_only']
    TT_LABELS = ['N→W', 'W→N', 'W→W', 'N→N']
    _VLINE_COLORS = ['#555555', '#888888']
    if timing is not None:
        init_end  = int(timing['init_steps'])                               # e.g. 10
        delay_end = int(timing['init_steps']) + int(timing['delay_steps'])  # e.g. 32
        phase_vlines = [init_end, delay_end]
        phase_labels = [f'Init ({init_end})', f'Delay ({delay_end})']
    else:
        phase_vlines = []
        phase_labels = []
  
    # find intact trajectories
    intact = None
    if 'intact' in trajs:
        intact = trajs['intact']
    else:
        for wn_data in trajs.values():
            if isinstance(wn_data, dict) and 'intact' in wn_data:
                intact = wn_data['intact']
                break
    if intact is None:
        print('  No intact trajectories found; skipping.')
        return None

    def _trial_mean_local(arrs):
        if not arrs:
            return None
        L = min(a.shape[0] - 1 for a in arrs)
        if L < 1:
            return None
        return np.stack([a[1:L + 1] for a in arrs]).mean(0)

    means = {tt: _trial_mean_local(intact.get(tt, [])) for tt in TT_ORDER}
    valid = [(tt, lab) for tt, lab in zip(TT_ORDER, TT_LABELS)
             if means[tt] is not None]
    if not valid:
        print('  No valid data; skipping.')
        return None

    # Symmetric color limits from raw values (percentile-clipped to avoid
    # outlier wash-out, which otherwise hides the sequential diagonal)
    all_vals = np.concatenate([means[tt].ravel() for tt, _ in valid])
    vlim = max(abs(np.percentile(all_vals, 2)),
               abs(np.percentile(all_vals, 98)), 1e-8)

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
        mat = means[tt].T  # (N, T)
        H, T = mat.shape

        # Sort neurons by peak time → sequential diagonal
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
            ax.set_ylabel('Neuron ID (sorted by peak time)',
                          fontsize=FONT['label'])
        
        xticks = sorted(set(
            [0] + [v for v in phase_vlines if 0 < v < T] + [T - 1]
        ))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks], fontsize=7)

        for vi, (vx, vlabel, vc) in enumerate(
                zip(phase_vlines, phase_labels, _VLINE_COLORS)):
            if 0 < vx < T:
                ax.axvline(vx, color=vc, ls='--', lw=0.9, alpha=0.70,
                           zorder=5)

        # Diagonal reference line to highlight sequential structure
        ax.plot([0, T - 1], [0, H - 1],
                color='white', ls=':', lw=0.7, alpha=0.45)

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
#  Master summary figure
# ═══════════════════════════════════════════════════════════════════

def make_summary_figure(seed, config,
                        X_nw, X_wn,
                        Y_nw_t, Y_wn_t, X_recon,
                        sort_nw, sort_wn,
                        test_nw, test_wn, k_chosen,
                        landscape, quiver,
                        proj_trajs, error_trajs,
                        centers, timing, L):
    set_publication_style()
    fig = plt.figure(figsize=(22, 9.0))
    gs = gridspec.GridSpec(
        2, 5, width_ratios=[1, 1, 1.10, 1, 1.55],
        wspace=0.55, hspace=0.45, figure=fig)

    X_nw_z   = zscore_columns(X_nw)
    X_wn_z   = zscore_columns(X_wn)
    Xrec_nw  = X_recon[:L]; Xrec_wn = X_recon[L:L * 2]
    Xrec_nw_z = zscore_columns(Xrec_nw)
    Xrec_wn_z = zscore_columns(Xrec_wn)
    Y_nw_n = Y_nw_t / (Y_nw_t.max() + 1e-9)
    Y_wn_n = Y_wn_t / (Y_wn_t.max() + 1e-9)

    # ── Row 0 : N→W ───────────────────────────────────────────────
    _plot_heatmap(fig.add_subplot(gs[0, 0]), X_nw_z,
                  'N→W: Input data', sort_nw,
                  vmin=-2.5, vmax=2.5, show_xlabel=False)
    _plot_heatmap(fig.add_subplot(gs[0, 1]), Y_nw_n,
                  'N→W: Supervised data', sort_nw,
                  cmap=CMAP_TMPL, vmin=0, vmax=1,
                  show_ylabel=False, show_xlabel=False)
    _plot_dropout_single(
        fig.add_subplot(gs[0, 2]), test_nw, k_chosen,
        title='N→W: Dimension dropout test',
        show_xlabel=False, show_arrow=True)
    _plot_heatmap(fig.add_subplot(gs[0, 3]), Xrec_nw_z,
                  'N→W: Reconstructed data', sort_nw,
                  vmin=-2.5, vmax=2.5,
                  show_ylabel=False, show_xlabel=False)
    _plot_pls_2d(
        fig.add_subplot(gs[0, 4]),
        flow=dict(landscape, **quiver),
        proj_intact=proj_trajs.get('intact', {}).get('nrem_to_wake', []),
        proj_error=error_trajs.get('nrem_to_wake', []),
        tt='nrem_to_wake', timing=timing, centers=centers,
        show_legend=True)

    # ── Row 1 : W→N ───────────────────────────────────────────────
    _plot_heatmap(fig.add_subplot(gs[1, 0]), X_wn_z,
                  'W→N: Input data', sort_wn,
                  vmin=-2.5, vmax=2.5)
    _plot_heatmap(fig.add_subplot(gs[1, 1]), Y_wn_n,
                  'W→N: Supervised data', sort_wn,
                  cmap=CMAP_TMPL, vmin=0, vmax=1, show_ylabel=False)
    _plot_dropout_single(
        fig.add_subplot(gs[1, 2]), test_wn, k_chosen,
        title='W→N: Dimension dropout test',
        show_xlabel=True, show_arrow=True)
    _plot_heatmap(fig.add_subplot(gs[1, 3]), Xrec_wn_z,
                  'W→N: Reconstructed data', sort_wn,
                  vmin=-2.5, vmax=2.5, show_ylabel=False)
    _plot_pls_2d(
        fig.add_subplot(gs[1, 4]),
        flow=dict(landscape, **quiver),
        proj_intact=proj_trajs.get('intact', {}).get('wake_to_nrem', []),
        proj_error=error_trajs.get('wake_to_nrem', []),
        tt='wake_to_nrem', timing=timing, centers=centers,
        show_legend=True)

    fig.suptitle(
        f'Sequence-Supervised PLSR  │  Seed {seed}  │  '
        f'Joint NW+WN, per-condition sort  │  K={k_chosen}  │  '
        f'σ={TEMPLATE_SIGMA}',
        fontsize=14, fontweight='bold', y=1.01)

    sp = os.path.join(_fig_dir(config), f'plsr_summary_seed{seed}.png')
    save_figure(fig, sp); plt.close(fig)
    print(f"  ✓ saved: {sp}")


# ═══════════════════════════════════════════════════════════════════
#  Per-seed orchestrator
# ═══════════════════════════════════════════════════════════════════

def analyse_seed_plsr(seed, model, data_loader, neuron_char,
                      config, device, force_rerun=False):
    cdir = _cache_dir(config)

    cache_t = os.path.join(cdir, f'seed{seed}_trajs.pkl')
    cached = None if force_rerun else _load(cache_t)
    if cached is not None and isinstance(cached, tuple) and len(cached) == 3:
        trajs, timing, pred = cached
        print("  trajectories loaded from cache")
    else:
        print("  collecting trajectories ...")
        trajs, timing, pred = collect_dynamics_trajectories(
            model, data_loader, neuron_char, device,
            ablation_window='init_delay')
        _save((trajs, timing, pred), cache_t)

    intact = trajs.get('intact', {})
    arr_nw = intact.get('nrem_to_wake', [])
    arr_wn = intact.get('wake_to_nrem', [])
    X_nw = _trial_mean(arr_nw, skip=1)
    X_wn = _trial_mean(arr_wn, skip=1)
    if X_nw is None or X_wn is None:
        print("  ✗ missing transition data — abort"); return None

    L = min(X_nw.shape[0], X_wn.shape[0])
    X_nw, X_wn = X_nw[:L], X_wn[:L]

    Y_nw, peaks_nw, sort_nw = build_sequential_template(X_nw)
    Y_wn, peaks_wn, sort_wn = build_sequential_template(X_wn)
    X_joint = np.vstack([X_nw, X_wn])
    Y_joint = np.vstack([Y_nw,  Y_wn])

    print(f"  X_joint shape: {X_joint.shape}  (NW len={L}, WN len={L})")

    # ── dropout tests ──
    print(f"  Dropout test  (N→W)  k=1..{MAX_PLS_COMPONENTS} ...")
    test_nw    = dimension_dropout_test(
        X_nw, Y_nw, MAX_PLS_COMPONENTS, desc='  PLS dropout NW')
    print(f"  Dropout test  (W→N)  k=1..{MAX_PLS_COMPONENTS} ...")
    test_wn    = dimension_dropout_test(
        X_wn, Y_wn, MAX_PLS_COMPONENTS, desc='  PLS dropout WN')
    print(f"  Dropout test (joint)  k=1..{MAX_PLS_COMPONENTS} ...")
    test_joint = dimension_dropout_test(
        X_joint, Y_joint, MAX_PLS_COMPONENTS, desc='  PLS dropout JOINT')
    k_chosen   = select_k_components(test_joint)
    print(f"  → K={k_chosen}  | "
          f"diss(joint)={test_joint['dissimilarity'][k_chosen-1]:.3f}  | "
          f"cum-var(joint)="
          f"{test_joint['variance_explained_cum'][k_chosen-1]:.1f}%")

    # ── fit on joint data ──
    pls = fit_plsr(X_joint, Y_joint, k=k_chosen)
    X_scores = pls.transform(X_joint)
    X_recon  = pls_inverse(X_scores, pls)

    # ── error trials projected to PLS basis ──
    TT_TO_ABL = {
        'nrem_to_wake': 'ablate_n2w',
        'wake_to_nrem': 'ablate_w2n',
    }
    print("  Filtering error trials & projecting into PLS basis ...")
    error_trajs: Dict[str, list] = {}
    for tt, abl_key in TT_TO_ABL.items():
        abl_arrs  = trajs.get(abl_key, {}).get(tt, [])
        abl_preds = pred.get(abl_key, {}).get(tt, []) if pred else []
        if not abl_arrs or not abl_preds:
            error_trajs[tt] = []
            print(f"    {abl_key}/{tt}: no trials available")
            continue
        err_arrs, n_err, n_tot = filter_error_trials(
            abl_arrs, abl_preds, tt)
        print(f"    {abl_key}/{tt}: {n_err}/{n_tot} error trials kept")
        error_trajs[tt] = [pls.transform(a) for a in err_arrs]

    print("  Projecting all trajectories into PLS basis ...")
    proj_trajs = project_trajs(trajs, pls)

    pls_range = _get_pls_range(proj_trajs)
    print(f"  PLS range: PLS1 [{pls_range[0]:.2f}, {pls_range[1]:.2f}]  "
          f"PLS2 [{pls_range[2]:.2f}, {pls_range[3]:.2f}]")

    print("  Landscape (zero-input window only) ...")
    landscape = compute_landscape_pls(
        model, pls, pls_range, device,
        trajs=trajs, proj_trajs=proj_trajs,
        timing=timing,
        grid_res=GRID_RES)
    print("  Quiver (zero-input window only) ...")
    quiver = compute_quiver_pls(
        model, pls, pls_range, device,
        trajs=trajs, proj_trajs=proj_trajs,
        timing=timing,
        n_quiver=N_QUIVER, n_steps=QUIVER_N_STEPS)

    print("  Computing PLS-space attractor centres ...")
    centers = _compute_pls_centers(intact, pls)

    # ── summary figure ──
    make_summary_figure(seed, config, X_nw, X_wn,
                        Y_nw, Y_wn, X_recon,
                        sort_nw, sort_wn,
                        test_nw, test_wn, k_chosen,
                        landscape, quiver,
                        proj_trajs, error_trajs,
                        centers=centers,
                        timing=timing, L=L)

    # ── extended trajectory figure ──
    print("\n  Generating PLS extended trajectory figure ...")
    generate_pls_extended_figure(
        seed=seed, config=config, model=model,
        pls=pls, landscape=landscape, quiver=quiver,
        trajs=trajs, pred_states=pred,
        proj_trajs=proj_trajs,
        timing=timing, centers=centers,
        device=device,
        n_extra_steps=EXTEND_N_STEPS,
    )

    # ── 4-types heatmap (z-scored) ──
    print("\n  Generating intact 4-types heatmap (z-scored) ...")
    plot_4_types_heatmaps(
        trajs=trajs,
        save_dir=os.path.join(_fig_dir(config), 'intact_activity'),
        seed=seed,
        timing=timing
    )


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def _safe_load_neuron_char(config, seed):
    res = _load_neuron_char_from_cache(config, seed)
    if res is None:
        return None
    if isinstance(res, tuple):
        res = res[0]
    if not isinstance(res, NeuronCharacterization):
        return None
    return res


def main():
    parser = argparse.ArgumentParser(
        description='PLSR sequence-supervised dynamics analysis')
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--all_seeds', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--force_rerun', action='store_true')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    args = parser.parse_args()

    config = get_default_config(gpu_id=args.gpu)
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.hidden_dim:
        config.model.hidden_dim = args.hidden_dim
    print_config(config)

    from train import load_trained_models
    print("\nLoading models ...")
    models = load_trained_models(config)
    if not models:
        print("ERROR: no models"); return

    if args.all_seeds:
        seeds = sorted(models.keys())
    elif args.seeds:
        seeds = [s for s in args.seeds if s in models]
    else:
        seeds = [sorted(models.keys())[0]]

    from task import load_datasets, create_dataloaders
    bundle = load_datasets(config.paths)
    loaders = create_dataloaders(
        bundle, batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        pin_memory=config.train.pin_memory)

    neuron_chars = {}
    for s in seeds:
        nc = _safe_load_neuron_char(config, s)
        if nc is not None:
            neuron_chars[s] = nc
    if not neuron_chars:
        print("ERROR: no neuron chars"); return

    print("\n" + "═" * 68)
    print("  Sequence-Supervised PLSR Dynamics")
    print(f"  Joint fit on N→W + W→N concatenated  │  σ={TEMPLATE_SIGMA}")
    print(f"  Per-condition dropout panels (reference-figure C style)")
    print(f"  Quiver: {QUIVER_N_STEPS}-step displacement, "
          f"const-z PLS-3+ slice")
    print(f"  Extended trajectories: +{EXTEND_N_STEPS} autonomous steps")
    print(f"  4-types heatmap: z-scored per neuron")
    print("═" * 68)

    for seed in seeds:
        if seed not in models or seed not in neuron_chars:
            continue
        print(f"\n{'─' * 50}\n  Seed {seed}\n{'─' * 50}")
        analyse_seed_plsr(
            seed=seed, model=models[seed],
            data_loader=loaders['ablation'],
            neuron_char=neuron_chars[seed],
            config=config, device=config.device,
            force_rerun=args.force_rerun)

    print("\nDone.")


if __name__ == "__main__":
    main()