# NP_State_trans

Code associated with the manuscript **"Distinct spiking sequences mediate global brain state transitions"**.

This repository contains the main analysis scripts used for Neuropixels global brain state transition analyses and related RNN modeling. The code is organized as research-analysis scripts rather than a single turnkey package; most scripts require setting the subject ID and local data paths near the top of each file before running.

## Repository Contents

### Neuropixels / brain-state analysis

- `ks_postprocessing.m`
  Kilosort post-processing, single-unit quality control, spike binning, and preparation of `sxx_postprocess_0.5s.mat`.

- `identify_state_modulated_neurons.m`
  Identification of state-modulated neurons around REM, wake, and NREM transitions. Results are appended to the post-processed subject file.

- `SVMforCorrectTransPoint_GPRforBoundarySeeking.m`
  Within-session transition alignment and initial sequence extraction using SVM/GPR. This generates intermediate `sxx_SVM.mat` and `sxx_SVM_wake.mat` files.

- `GPR_find_seq_in_all_trials.m` and `GPR_for_time_decoding.m`
  GPR-based transition-time decoding and sequence detection across trials. These scripts generate or use `sxx_SVM2.mat` and `sxx_SVM_wake2.mat`, which are the main downstream sequence-analysis files.

- `delta_alignment_for_br_t_sort.m`
  Aligns sequence timing to delta-power dynamics and appends delta-alignment variables to the sequence files.

- `Focality_cal.m` and `participation_degree_statis.m`
  Brain-region distribution analyses for sequence neurons, including focality and participation-degree statistics.

- `attractor_pca_analyze.m`, `PLSregression_for_whole_dynamics.m`, and `Parallel_analysis.m`
  PCA/PLS-based analyses of transition dynamics, attractor structure, and dimensionality.

- `transition_probability_optogenetics.m`
  Transition-probability analysis for optogenetic experiments.

### Autoencoder / bridge-path analyses

- `attractor_autoencoder.py`
  Autoencoder modeling of attractor manifolds from transition-related neural trajectories.

- `WN_s74_bridge_path.py` and `NW_s74_bridge_path.py`
  Example bridge-path autoencoder analyses for wake-to-NREM and NREM-to-wake transitions.

### RNN modeling

The `RNN_Modeling_YZ/` folder contains the continuous-time RNN model used to simulate sleep-wake state maintenance and transition tasks.

Main components:

- `task.py`: paired sleep-wake trial generation.
- `model.py`: continuous-time RNN model.
- `train.py`: training and evaluation.
- `analysis_neurons.py`, `analysis_perturbation.py`, `analysis_statistics.py`: neuron characterization, ablation, perturbation, and statistical analysis.
- `vis_ablation.py`, `vis_pca_dynamics.py`, `vis_plsr_dynamics.py`, `vis_perturbation.py`: figure-generation scripts.
- `run_experiment.py`: main entry point for data generation, training, analysis, perturbation, statistics, and visualization.

See `RNN_Modeling_YZ/README.md` for the RNN-specific environment and commands.

## Basic Workflow

The main experimental-analysis workflow is:

1. Run Kilosort post-processing and single-unit quality control.
2. Identify state-modulated neurons and prepare subject-level post-processing files.
3. Align sleep-wake transitions with SVM/GPR and extract sequence neurons.
4. Refine transition timing and sequence detection across all trials.
5. Analyze brain-region distribution, sequence timing, dimensionality, attractor structure, and optogenetic transition probability.
6. Use the RNN modeling code for complementary mechanistic simulations and perturbation analyses.

## Notes

- Many scripts contain hard-coded local paths such as `X:\...`, `Y:\...`, or `~\...`; update these paths before running in a new environment.
- Several MATLAB scripts rely on helper functions and plotting utilities from the local MATLAB code path, for example `stable_states`, `trainRegressionModel`, and custom colormaps.
- The top-level MATLAB scripts are written mainly as analysis records for the paper figures, so subject IDs and transition direction settings may need to be edited manually.
- The full raw and processed datasets are not fully contained in this repository now; the scripts expect preprocessed `.mat` files such as `sxx_postprocess_0.5s.mat`, `sxx_SVM2.mat`, and `sxx_SVM_wake2.mat`.
- `mice_example_data/`, large `.mat` data files, and generated experiment outputs are kept out of Git; store them separately or use Git LFS if they need to be shared.
