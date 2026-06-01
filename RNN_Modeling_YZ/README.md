# Sleep-Wake State Transition in Recurrent Neural Networks

This project trains continuous-time RNNs to perform a sleep-wake state maintenance and transition task, then systematically analyzes the learned dynamics through ablation, perturbation, flow-field analysis, and neuron-level functional characterization.

## Hardware

All experiments were conducted on a machine equipped with an **Intel® Xeon® Gold 6530** CPU.

## Environment Setup

```bash
conda env create -f rnn.yml
conda activate rnn
```

## Pipeline 
```
data ──▶ train ──▶ analyze ──▶ perturb ──▶ stats ──▶ visualize
```

## Run

```bash
# for training
python run_experiment.py --step data

python run_experiment.py --step train --gpu 0

# for evaluation and analysis
python run_experiment.py --step analyze

python run_experiment.py --step perturb  

python run_experiment.py --step stats 

python run_experiment.py --step visualize 

python vis_pca_dynamics.py --all_seeds

python vis_plsr_dynamics.py --all_seeds

```

关于结果，gigures_init+delay和figures/perturbation 均由全部20个model计算所得。