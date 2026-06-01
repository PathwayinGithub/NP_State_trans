import os
import torch
import numpy as np
import math
from typing import Dict, List, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import dataclass, field

from analysis_core import (
    TRIAL_TYPES, CONDITIONS, EXPECTED_STATE_MAP,
    classify_trial_state, resolve_resp_start,
)
from analysis_neurons import (
    NeuronCharacterization,
    build_neuron_ablation_conditions,
)


# ======================================================================
#  Constants
# ======================================================================

NOISE_WINDOW_DEFS = {
    # 'delay':      'Delay Full',
    'init_delay':  'Init + Delay',
}

DEFAULT_NOISE_WINDOWS = ['init_delay']#'delay',

ABLATION_WINDOW_DEFS = {
    'init_delay':  'Init + Delay',
}

DEFAULT_ABLATION_WINDOWS = ['init_delay']

DEFAULT_PERTURBATION_STRENGTHS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


# ======================================================================
#  Data-class
# ======================================================================

@dataclass
class PerturbationResult:
    state_counts: Dict
    perturbation_strengths: List[float]
    perturbation_timestep: int
    summary: Dict = field(default_factory=dict)
    n_models: int = 1
    accuracy_per_model: Dict = field(default_factory=dict)
    noise_window: str = ''
    abl_window: str = ''
    noise_start: int = 0
    noise_end: int = 0
    abl_start: int = 0
    abl_end: int = 0

    def compute_summary(self):
        for trial_type in TRIAL_TYPES:
            self.summary[trial_type] = {}
            expected = EXPECTED_STATE_MAP[trial_type]
            for condition in self.state_counts.get(trial_type, {}):
                self.summary[trial_type][condition] = {}
                for strength in self.perturbation_strengths:
                    counts = self.state_counts[trial_type][condition].get(
                        strength)
                    if counts is None:
                        self.summary[trial_type][condition][strength] = {
                            'accuracy': 0, 'accuracy_std': 0,
                            'wake_rate': 0, 'nrem_rate': 0, 'total': 0,
                        }
                        continue
                    total = sum(counts.values())
                    if total == 0:
                        self.summary[trial_type][condition][strength] = {
                            'accuracy': 0, 'accuracy_std': 0,
                            'wake_rate': 0, 'nrem_rate': 0, 'total': 0,
                        }
                        continue

                    accuracy = counts[expected] / total
                    acc_std = 0.0

                    if (self.accuracy_per_model
                            and trial_type in self.accuracy_per_model
                            and condition in self.accuracy_per_model[
                                trial_type]
                            and strength in self.accuracy_per_model[
                                trial_type][condition]):
                        al = self.accuracy_per_model[trial_type][
                            condition][strength]
                        if len(al) > 1:
                            acc_std = float(np.std(al))

                    self.summary[trial_type][condition][strength] = {
                        'accuracy': accuracy,
                        'accuracy_std': acc_std,
                        'wake_rate': counts['wake'] / total,
                        'nrem_rate': counts['nrem'] / total,
                        'total': total,
                    }

# ======================================================================
#  Window bounds helpers
# ======================================================================

def _noise_window_bounds(window_name, init_steps, delay_steps,
                         total_steps):
    delay_start = init_steps
    delay_end = init_steps + delay_steps
    resp_start = delay_end
    _MAP = {
        'full_trial':     (0, total_steps),
        'init':           (0, init_steps),
        'delay':          (delay_start, delay_end),
        'init_delay':     (0, delay_end),
        'response':       (resp_start, total_steps),
        'delay_response': (delay_start, total_steps)
    }
    return _MAP.get(window_name, (delay_start, delay_end))


def _ablation_window_bounds(window_name, init_steps, resp_start,
                            total_steps):
    _MAP = {
        'full_trial':     (0, total_steps),
        'init':           (0, init_steps),
        'delay':          (init_steps, resp_start),
        'init_delay':     (0, resp_start),
        'response':       (resp_start, total_steps),
        'delay_response': (init_steps, total_steps),
    }
    return _MAP.get(window_name, (0, total_steps))


# ======================================================================
#  Single-model perturbation
# ======================================================================

def run_perturbation_single(
    model,
    data_loader: DataLoader,
    config,
    device: str,
    perturbation_strengths: List[float],
    neuron_char: Optional[NeuronCharacterization] = None,
    *,
    noise_start: Optional[int] = None,
    noise_end: Optional[int] = None,
    abl_start: int = 0,
    abl_end: Optional[int] = None,
) -> PerturbationResult:
    model.eval()
    if 0.0 not in perturbation_strengths:
        perturbation_strengths = sorted(
            set([0.0] + list(perturbation_strengths)))

    hidden_dim = model.hidden_dim
    if neuron_char is not None:
        conditions = build_neuron_ablation_conditions(
            neuron_char, hidden_dim, device)
    else:
        conditions = {c: (None, None) for c in CONDITIONS}

    state_counts = {
        trial_type: {
            cond_name: {
                strength: {'wake': 0, 'nrem': 0}
                for strength in perturbation_strengths
            }
            for cond_name in conditions
        }
        for trial_type in TRIAL_TYPES
    }

    timing_resolved = False
    resp_start = noise_s = noise_e = abl_e_resolved = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Perturbation', leave=False):
            trial1_inputs = batch['trial1']['inputs'].to(device)
            trial2_inputs = batch['trial2']['inputs'].to(device)
            trial2_targets = batch['trial2']['targets'].cpu().numpy()
            metadata = batch['metadata']
            batch_size = trial1_inputs.shape[0]
            seq_len = trial2_inputs.shape[1]

            if not timing_resolved:
                init_steps, delay_steps, _, resp_start = \
                    resolve_resp_start(metadata[0])
                noise_s = (noise_start if noise_start is not None
                           else init_steps)
                noise_e = (noise_end if noise_end is not None
                           else init_steps + delay_steps)
                abl_e_resolved = (abl_end if abl_end is not None
                                  else seq_len)
                timing_resolved = True

            _, hidden1 = model(trial1_inputs, h0=None, return_hidden=True)
            h_final_trial1 = hidden1[:, -1, :]

            for strength in perturbation_strengths:
                for cond_name, (afn, adir) in conditions.items():
                    h = h_final_trial1.clone()
                    step_outputs = []

                    for t in range(seq_len):
                        x_t = trial2_inputs[:, t, :]
                        abl_active = (afn is not None
                                      and abl_start <= t < abl_e_resolved)

                        output_t, h = model.single_step(
                            x_t, h,
                            ablation_fn=afn if abl_active else None,
                            ablation_direction=adir if abl_active
                            else None)
                        step_outputs.append(output_t)

                        if noise_s <= t < noise_e:
                            noise_window_len = noise_e - noise_s
                            sigma = strength / math.sqrt(noise_window_len)
                            h = h + torch.randn_like(h) * sigma

                            if abl_active and afn is not None: #! we need make sure the silencing is still working.
                                h = afn(h, h, adir)

                    outputs2 = torch.stack(step_outputs, dim=1)
                    outputs2_np = outputs2.cpu().numpy()
                    for b in range(batch_size):
                        trial_type = metadata[b]['trial_type']
                        predicted = classify_trial_state(
                            outputs2_np[b, resp_start:],
                            trial2_targets[b, resp_start:])
                        state_counts[trial_type][cond_name][strength][
                            predicted] += 1

    result = PerturbationResult(
        state_counts=state_counts,
        perturbation_strengths=perturbation_strengths,
        perturbation_timestep=-1, n_models=1,
        noise_start=noise_s, noise_end=noise_e,
        abl_start=abl_start, abl_end=abl_e_resolved)
    result.compute_summary()
    return result


# ======================================================================
#  Multi-model perturbation
# ======================================================================

def run_perturbation_multi(
    models: Dict,
    data_loader: DataLoader,
    config,
    device: str,
    perturbation_strengths: Optional[List[float]] = None,
    neuron_chars: Optional[Dict] = None,
    *,
    noise_start: Optional[int] = None,
    noise_end: Optional[int] = None,
    abl_start: int = 0,
    abl_end: Optional[int] = None,
) -> PerturbationResult:

    if perturbation_strengths is None:
        perturbation_strengths = list(DEFAULT_PERTURBATION_STRENGTHS)

    if 0.0 not in perturbation_strengths:
        perturbation_strengths = sorted(
            set([0.0] + list(perturbation_strengths)))

    condition_set = set(CONDITIONS)
    if neuron_chars:
        for nc in neuron_chars.values():
            if nc is not None:
                _tmp = build_neuron_ablation_conditions(
                    nc, list(models.values())[0].hidden_dim, device)
                condition_set.update(_tmp.keys())
                break
    condition_list = sorted(condition_set)

    accuracy_per_model = {
        tt: {c: {s: [] for s in perturbation_strengths}
             for c in condition_list}
        for tt in TRIAL_TYPES
    }
    total_state_counts = {
        tt: {c: {s: {'wake': 0, 'nrem': 0}
                 for s in perturbation_strengths}
             for c in condition_list}
        for tt in TRIAL_TYPES
    }

    for seed, model in models.items():
        nc = neuron_chars.get(seed) if neuron_chars else None
        print(f"    seed={seed}")

        single_result = run_perturbation_single(
            model, data_loader, config, device,
            perturbation_strengths, neuron_char=nc,
            noise_start=noise_start, noise_end=noise_end,
            abl_start=abl_start, abl_end=abl_end)

        for trial_type in TRIAL_TYPES:
            for cond_name in single_result.state_counts[trial_type]:
                if cond_name not in total_state_counts[trial_type]:
                    continue
                for strength in perturbation_strengths:
                    sc = single_result.state_counts[trial_type][cond_name]
                    if strength not in sc:
                        continue
                    for state in ('wake', 'nrem'):
                        total_state_counts[trial_type][cond_name][
                            strength][state] += sc[strength][state]
                    sm = single_result.summary[trial_type].get(
                        cond_name, {}).get(strength, {})
                    accuracy_per_model[trial_type][cond_name][
                        strength].append(sm.get('accuracy', 0.0))

    result = PerturbationResult(
        state_counts=total_state_counts,
        perturbation_strengths=perturbation_strengths,
        perturbation_timestep=0,
        n_models=len(models),
        accuracy_per_model=accuracy_per_model,
        noise_start=noise_start if noise_start is not None else 0,
        noise_end=noise_end if noise_end is not None else 0,
        abl_start=abl_start,
        abl_end=abl_end if abl_end is not None else 0)
    result.compute_summary()
    return result


# ======================================================================
#  Per-noise-window driver
# ======================================================================

def run_perturbation_by_noise_window(
    models, data_loader, config, device,
    perturbation_strengths=None,
    neuron_chars=None,
    noise_windows=None, abl_window='init_delay',
) -> Dict[str, PerturbationResult]:

    if noise_windows is None:
        noise_windows = list(DEFAULT_NOISE_WINDOWS)
    if perturbation_strengths is None:
        perturbation_strengths = list(DEFAULT_PERTURBATION_STRENGTHS)

    for batch in data_loader:
        init_steps, delay_steps, _, resp_start = resolve_resp_start(
            batch['metadata'][0])
        total_steps = batch['trial2']['inputs'].shape[1]
        break

    abl_s, abl_e = _ablation_window_bounds(
        abl_window, init_steps, resp_start, total_steps)
    abl_label = ABLATION_WINDOW_DEFS.get(abl_window, abl_window)

    results = {}
    for noise_window in noise_windows:
        ns, ne = _noise_window_bounds(noise_window, init_steps,
                                      delay_steps, total_steps)
        noise_label = NOISE_WINDOW_DEFS.get(noise_window, noise_window)

        print(f"\n  Ablation: {abl_label} (steps {abl_s}–{abl_e})  |  "
              f"Noise: {noise_label} (steps {ns}–{ne})")

        perturbation_result = run_perturbation_multi(
            models=models, data_loader=data_loader,
            config=config, device=device,
            perturbation_strengths=perturbation_strengths,
            neuron_chars=neuron_chars,
            noise_start=ns, noise_end=ne,
            abl_start=abl_s, abl_end=abl_e)
        perturbation_result.noise_window = noise_window
        perturbation_result.abl_window = abl_window
        perturbation_result.noise_start = ns
        perturbation_result.noise_end = ne
        perturbation_result.abl_start = abl_s
        perturbation_result.abl_end = abl_e
        results[noise_window] = perturbation_result

    return results


# ======================================================================
#  Full-grid driver
# ======================================================================

def run_perturbation_all_windows(
    models, data_loader, config, device,
    perturbation_strengths=None,
    neuron_chars=None,
    noise_windows=None, ablation_windows=None,
) -> Dict[str, Dict[str, PerturbationResult]]:
    """Run full perturbation grid:
    ablation windows × noise windows × pathway conditions."""

    if ablation_windows is None:
        ablation_windows = list(DEFAULT_ABLATION_WINDOWS)
    if noise_windows is None:
        noise_windows = list(DEFAULT_NOISE_WINDOWS)

    all_results = {}
    for abl_window in ablation_windows:
        abl_label = ABLATION_WINDOW_DEFS.get(abl_window, abl_window)
        print(f"\n{'#' * 65}")
        print(f"  ABLATION WINDOW: {abl_label}")
        print(f"{'#' * 65}")

        all_results[abl_window] = run_perturbation_by_noise_window(
            models=models, data_loader=data_loader,
            config=config, device=device,
            perturbation_strengths=perturbation_strengths,
            neuron_chars=neuron_chars,
            noise_windows=noise_windows,
            abl_window=abl_window)

    n_total = len(ablation_windows) * len(noise_windows)
    print(f"\n  Perturbation grid complete: "
          f"{len(ablation_windows)} abl × {len(noise_windows)} noise "
          f"= {n_total} conditions")
    return all_results