# NP_State_trans

Code associated with the manuscript **"Distinct spiking sequences mediate global brain state transitions"**.

This repository contains the main analysis scripts used for Neuropixels global brain state transition analyses and related RNN modeling. The code is organized as research-analysis scripts rather than a single turnkey package; most scripts require setting the subject ID and local data paths near the top of each file before running.

The single-mouse example is the main-figure example mouse `s74`, corresponding to `miceID240428` in Table S2. Example data are not stored in this repository because of GitHub file-size limits; but can be downloaded from zenodo: **https://doi.org/10.5281/zenodo.20525628** . In the scripts, paths such as `~/s74_SVM2.mat`, `~/s74_postprocess_0.5s.mat`, and `~/s74_WN_bridge_path_AEmodel.pth` indicate the expected final file names after download. Replace `~/` with the local folder that contains those files.

Trained Autoencoder and RNN weights are provided and can be loaded directly. `ks_postprocessing.m` is included for reference and does not need to be rerun for the example workflow, because the postprocessed output is provided. Multi-animal statistical analyses are included for reference, but they are not currently reproducible from the single-mouse example data.

Any other raw data are available from the corresponding author upon request.

Colormaps in the repo https://github.com/slandarer/slanColor were used.
