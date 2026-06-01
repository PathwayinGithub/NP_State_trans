import argparse
import os
import torch
import numpy as np

from config import get_default_config, print_config
from task import generate_and_save_datasets, load_datasets, create_dataloaders
from train import train_multiple_seeds, load_trained_models

from analysis_core import (
    TRIAL_TYPES,
    resolve_resp_start,
    save_pickle, load_pickle,
)
from analysis_neurons import (
    collect_hidden_trajectories,
    characterize_neurons,
    run_temporal_ablation_multi,
    collect_condition_trajectories,
    TEMPORAL_WINDOW_LABELS,
)
from analysis_statistics import (
    compute_statistics_for_window,
    generate_statistics_text_report,
)
from analysis_perturbation import run_perturbation_all_windows


# =====================================================================
#  Constants
# =====================================================================

# Only one ablation window for temporal & gradual ablation
ABLATION_WINDOWS = ['init_delay']

# Only one trajectory window (same as ablation)
TRAJECTORY_WINDOWS = ['init_delay']

# Perturbation: init_delay ablation × 3 noise windows
PERTURBATION_ABL_WINDOWS = ['init_delay']
PERTURBATION_NOISE_WINDOWS = ['delay']

GRADUAL_ALPHAS = [0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15,
                  0.18, 0.2, 0.4, 0.6, 0.8, 1.0]


# =====================================================================
#  Helpers
# =====================================================================

def resolve_timing_from_loader(data_loader):
    """Get temporal structure from first batch."""
    for batch in data_loader:
        metadata = batch['metadata']
        init_steps, delay_steps, _, _ = resolve_resp_start(metadata[0])
        total_steps = batch['trial2']['inputs'].shape[1]
        return init_steps, delay_steps, total_steps
    raise RuntimeError("Empty data loader")


def get_window_bounds(window_name, init_steps, delay_steps, total_steps):
    """Map window name to (start_step, end_step)."""
    resp_start = init_steps + delay_steps
    bounds = {
        'full_trial':     (0, total_steps),
        'init':           (0, init_steps),
        'delay':          (init_steps, resp_start),
        'init_delay':     (0, resp_start),
        'response':       (resp_start, total_steps),
        'delay_response': (init_steps, total_steps),
    }
    return bounds.get(window_name, (0, total_steps))


def get_trajectory_dir(config):
    """Canonical directory for on-disk trajectory pickles."""
    return os.path.join(
        config.paths.data_dir, 'analysis', 'temporal', 'trajectories')


def get_results_path(config):
    return os.path.join(config.paths.data_dir, 'experiment_results.pkl')


# =====================================================================
#  Step 1: Data Generation
# =====================================================================

def run_data_generation(config):
    print("\n" + "=" * 60)
    print("Step 1: Data Generation")
    print("=" * 60)
    return generate_and_save_datasets(config.task, config.paths, seed=42)


# =====================================================================
#  Step 2: Model Training
# =====================================================================

def run_training(config):
    print("\n" + "=" * 60)
    print("Step 2: Model Training")
    print("=" * 60)
    return train_multiple_seeds(config, verbose=True)


# =====================================================================
#  Step 3: Analysis
#    Phase A — per-model: threshold calibration + neuron characterisation
#    Phase B — temporal ablation across models (init_delay only)
#    Phase C — visualisation trajectory collection (init_delay only,
#              intact + all 6 ablation conditions, 400 per trial type)
# =====================================================================
def run_analysis(config, models):
    print("\n" + "=" * 60)
    print("Step 3: Analysis")
    print("=" * 60)

    bundle = load_datasets(config.paths)
    loaders = create_dataloaders(
        bundle,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        pin_memory=config.train.pin_memory,
    )
    data_loader = loaders['ablation']
    device = config.device

    init_steps, delay_steps, total_steps = resolve_timing_from_loader(
        data_loader)

    # ── Phase A: Per-model characterisation ──
    print("\n── Phase A: Per-model characterisation ──")
    neuron_characterizations = {}
    # directory for individual neuron characterisation files
    nc_save_dir = os.path.join(config.paths.data_dir, 'analysis')
    os.makedirs(nc_save_dir, exist_ok=True)

    for seed, model in models.items():
        print(f"\n  Model seed={seed}")
        if 'cuda' in device:
            torch.cuda.empty_cache()
        print("    Collecting hidden trajectories ...")
        trajectories, traj_init_steps, traj_delay_steps = \
            collect_hidden_trajectories(
                model, data_loader, device,
                n_samples=config.analysis.n_samples_for_characterization)

        print("    Characterising neurons ...")
        neuron_char = characterize_neurons(
            trajectories=trajectories,
            init_steps=traj_init_steps,
            delay_steps=traj_delay_steps,
            hidden_dim=model.hidden_dim,
            seed=seed,
        )
        neuron_characterizations[seed] = neuron_char

        # save individual file so vis_dynamics.py can find it
        nc_path = os.path.join(nc_save_dir, f'seed{seed}_neuron_char.pkl')
        save_pickle({
            'neuron_characterization': neuron_char,
        }, nc_path)

    # ── Phase B: Temporal ablation (init_delay only) ──
    print("\n── Phase B: Temporal ablation (init_delay) ──")
    temporal_ablation_result = run_temporal_ablation_multi(
        models=models,
        data_loader=data_loader,
        config=config.analysis,
        device=device,
        neuron_chars=neuron_characterizations,
        windows=ABLATION_WINDOWS,
    )
    print_temporal_summary(temporal_ablation_result)

    # ── Phase C: Visualisation trajectory collection (init_delay only) ──
    print("\n── Phase D: Visualisation trajectory collection ──")
    traj_dir = get_trajectory_dir(config)
    os.makedirs(traj_dir, exist_ok=True)

    for seed, model in models.items():
        nc = neuron_characterizations[seed]

        if 'cuda' in device:
            torch.cuda.empty_cache()

        for window_name in TRAJECTORY_WINDOWS:
            save_path = os.path.join(
                traj_dir, f'seed{seed}_{window_name}.pkl')

            # Skip if already exists (allows incremental runs)
            if os.path.exists(save_path):
                print(f"    seed={seed} / {window_name}: "
                      f"already on disk — skipped")
                continue

            start_step, end_step = get_window_bounds(
                window_name, init_steps, delay_steps, total_steps)
            print(f"    seed={seed} / {window_name} "
                  f"(ablation steps {start_step}–{end_step}) ...")

            cond_trajs = collect_condition_trajectories(
                model=model,
                data_loader=data_loader,
                device=device,
                neuron_char=nc,
                n_samples_per_type=config.analysis.n_samples_for_trajectories,
                ablation_start_step=start_step,
                ablation_end_step=end_step,
            )
            save_pickle(cond_trajs, save_path)

    # Count what was saved
    n_traj_files = len([f for f in os.listdir(traj_dir)
                        if f.endswith('.pkl')])
    print(f"\n  Trajectory files on disk: {n_traj_files}")

    return {
        'neuron_characterizations': neuron_characterizations,
        'temporal_ablation': temporal_ablation_result
    }

# =====================================================================
#  Step 4: Perturbation
#    Ablation: init_delay  ×  Noise: init, delay, init_delay
# =====================================================================

def run_perturbation(config, models, unified_results,
                     noise_windows=None, ablation_windows=None):
    print("\n" + "=" * 60)
    print("Step 4: Perturbation Analysis")
    print("=" * 60)

    # Use project defaults if not overridden by CLI
    if noise_windows is None:
        noise_windows = PERTURBATION_NOISE_WINDOWS
    if ablation_windows is None:
        ablation_windows = PERTURBATION_ABL_WINDOWS

    bundle = load_datasets(config.paths)
    loaders = create_dataloaders(
        bundle,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        pin_memory=config.train.pin_memory,
    )
    data_loader = loaders['ablation']
    device = config.device

    neuron_characterizations = unified_results['neuron_characterizations']

    perturbation_results = run_perturbation_all_windows(
        models=models,
        data_loader=data_loader,
        config=config,
        device=device,
        neuron_chars=neuron_characterizations,
        noise_windows=noise_windows,
        ablation_windows=ablation_windows,
    )

    n_abl = len(perturbation_results)
    n_noise = sum(len(v) for v in perturbation_results.values())
    print(f"\n  Perturbation complete: "
          f"{n_abl} abl × {n_noise // max(n_abl, 1)} noise windows")
    return perturbation_results


# =====================================================================
#  Step 5: Statistical Tests
# =====================================================================

def run_statistical_tests(config, unified_results):
    print("\n" + "=" * 60)
    print("Step 5: Statistical Tests")
    print("=" * 60)

    temporal_ablation_result = unified_results['temporal_ablation']

    all_statistics = {}
    for window_name in ABLATION_WINDOWS:
        stats = compute_statistics_for_window(
            temporal_ablation_result, window_name)
        if stats is not None:
            all_statistics[window_name] = stats
            report_path = os.path.join(
                config.paths.data_dir,
                f'statistics_report_{window_name}.txt')
            generate_statistics_text_report(stats, save_path=report_path)

    return all_statistics


# =====================================================================
#  Step 6: Visualization
# =====================================================================

def run_visualization(config, unified_results):
    """Generate all ablation and perturbation figures from cached results."""
    print("\n" + "=" * 60)
    print("Step 6: Visualization")
    print("=" * 60)

    from vis_ablation import generate_all_ablation_figures
    from vis_perturbation import generate_all_perturbation_figures

    # ── Build analysis_results dict expected by vis_ablation ──
    neuron_chars = unified_results.get('neuron_characterizations', {})

    class _SeedResultWrapper:
        def __init__(self, neuron_characterization):
            self.neuron_characterization = neuron_characterization

    individual_wrapped = {
        seed: _SeedResultWrapper(nc)
        for seed, nc in neuron_chars.items()
    }

    analysis_results = {
        'individual': individual_wrapped,
        'aggregate': {
            'n_models': len(neuron_chars),
        },
        'temporal': unified_results.get('temporal_ablation'),
    }
    
    # Use init_delay statistics for significance markers
    all_statistics = unified_results.get('statistics', {})
    primary_statistics = all_statistics.get('init_delay')
    if primary_statistics is None and all_statistics:
        primary_statistics = next(iter(all_statistics.values()))
    
    # ── A. Ablation figures ──
    print("\n── Ablation Figures ──")
    generate_all_ablation_figures(
        config=config,
        analysis_results=analysis_results,
        statistics=primary_statistics,
    )

    # ── B. Perturbation figures ──
    perturbation_results = unified_results.get('perturbation')
    if perturbation_results:
        print("\n── Perturbation Figures ──")
        generate_all_perturbation_figures(
            config=config,
            all_results=perturbation_results,
        )
    else:
        print("\n  No perturbation results — skipping perturbation figures")

    print("\n  All visualization complete.")


# =====================================================================
#  Printing
# =====================================================================

WINDOW_SHORT = {
    'init': 'Init', 'delay': 'Delay', 'init_delay': 'I+D',
    'full_trial': 'Full', 'response': 'Resp', 'delay_response': 'D+R',
}


def print_temporal_summary(temporal_result):
    if temporal_result is None:
        return
    print(f"\n{'=' * 70}")
    print("  TEMPORAL ABLATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Models: {temporal_result.n_models}  |  "
          f"Init={temporal_result.init_steps}  "
          f"Delay={temporal_result.delay_steps}  "
          f"Resp={temporal_result.response_steps}")

    key_pairs = [
        ('ablate_n2w', 'nrem_to_wake', 'Cut N→W', 'N→W'),
        ('ablate_w2n', 'wake_to_nrem', 'Cut W→N', 'W→N'),
        ('ablate_nn',  'nrem_only',    'Cut N→N', 'N Only'),
        ('ablate_ww',  'wake_only',    'Cut W→W', 'W Only'),
    ]

    header = f"  {'Condition':<12} {'Trial':<8}"
    for window_name in temporal_result.window_names:
        header += f" {WINDOW_SHORT.get(window_name, window_name[:6]):>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for condition, trial_type, cond_label, tt_label in key_pairs:
        row = f"  {cond_label:<12} {tt_label:<8}"
        for window_name in temporal_result.window_names:
            accuracy = temporal_result.accuracy_mean.get(
                window_name, {}).get(trial_type, {}).get(condition, 0)
            row += f" {accuracy:8.0%}"
        print(row)

    row_intact = f"\n  {'Intact':<12} {'(all)':<8}"
    for window_name in temporal_result.window_names:
        vals = [
            temporal_result.accuracy_mean.get(
                window_name, {}).get(trial_type, {}).get('intact', 0)
            for trial_type in TRIAL_TYPES
        ]
        row_intact += f" {np.mean(vals):8.0%}"
    print(row_intact)
    print("=" * 70)


# =====================================================================
#  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Sleep/Wake State Switching — Ablation Experiment')
    parser.add_argument('--step', type=str, default='all',
                        choices=['all', 'data', 'train', 'analyze',
                                 'perturb', 'stats', 'visualize'])
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--noise_windows', type=str, nargs='+',
                        default=None,
                        help='Noise windows for perturbation '
                             '(default: init, delay, init_delay)')
    parser.add_argument('--abl_windows', type=str, nargs='+',
                        default=None,
                        help='Ablation windows for perturbation '
                             '(default: init_delay)')
    parser.add_argument('--force_rerun', action='store_true',
                        help='Ignore cached results and rerun')
    args = parser.parse_args()

    config = get_default_config(gpu_id=args.gpu)
    if args.seeds:
        config.train.seeds = args.seeds
    if args.epochs:
        config.train.n_epochs = args.epochs
    if args.hidden_dim:
        config.model.hidden_dim = args.hidden_dim
    if args.batch_size:
        config.train.batch_size = args.batch_size
    print_config(config)

    step = args.step
    results_path = get_results_path(config)

    # ------------------------------------------------------------------
    #  Step 1: Data generation
    # ------------------------------------------------------------------
    if step in ('all', 'data'):
        data_path = config.paths.get_data_path('datasets.pkl')
        if not os.path.exists(data_path) or step == 'data':
            run_data_generation(config)

    # ------------------------------------------------------------------
    #  Step 2: Training
    # ------------------------------------------------------------------
    if step in ('all', 'train'):
        run_training(config)

    # ------------------------------------------------------------------
    #  Load models (needed for analysis / perturb)
    # ------------------------------------------------------------------
    models = None
    if step in ('all', 'analyze', 'perturb'):
        print("\nLoading trained models ...")
        models = load_trained_models(config)
        if not models:
            print("ERROR: No trained models. Run 'train' first.")
            return

    # ------------------------------------------------------------------
    #  Load or initialise unified results dict
    # ------------------------------------------------------------------
    unified_results = {}
    if not args.force_rerun and os.path.exists(results_path):
        cached = load_pickle(results_path)
        if cached is not None:
            unified_results = cached
            print(f"  Loaded cached results from {results_path}")

    # ------------------------------------------------------------------
    #  Step 3: Analysis
    # ------------------------------------------------------------------
    if step in ('all', 'analyze'):
        analysis_output = run_analysis(config, models)
        unified_results.update(analysis_output)
        unified_results['meta'] = {
            'n_models': len(models),
            'seeds': list(models.keys()),
            'windows': ABLATION_WINDOWS,
        }

    # Verify analysis data exists for downstream steps
    if step in ('perturb', 'stats', 'visualize'):
        if 'temporal_ablation' not in unified_results:
            print("ERROR: No analysis results found. "
                  "Run 'analyze' first.")
            return

    # ------------------------------------------------------------------
    #  Step 4: Perturbation
    # ------------------------------------------------------------------
    if step in ('all', 'perturb'):
        perturbation_results = run_perturbation(
            config, models, unified_results,
            noise_windows=args.noise_windows,
            ablation_windows=args.abl_windows,
        )
        unified_results['perturbation'] = perturbation_results

    # ------------------------------------------------------------------
    #  Step 5: Statistical tests
    # ------------------------------------------------------------------
    if step in ('all', 'stats'):
        statistics = run_statistical_tests(config, unified_results)
        unified_results['statistics'] = statistics

    # ------------------------------------------------------------------
    #  Step 6: Visualization
    # ------------------------------------------------------------------
    if step in ('all', 'visualize'):
        run_visualization(config, unified_results)

    # ------------------------------------------------------------------
    #  Save unified results
    # ------------------------------------------------------------------
    if step in ('all', 'analyze', 'perturb', 'stats'):
        save_pickle(unified_results, results_path)
        print(f"\n  Unified results saved: {results_path}")

    if step == 'all':
        print("\n" + "=" * 60)
        print("Experiment complete!")
        print("=" * 60)


if __name__ == "__main__":
    main()