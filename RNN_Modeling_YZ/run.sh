# # for training
python run_experiment.py --step data

python run_experiment.py --step train --gpu 0

# for evaluation and analysis
python run_experiment.py --step analyze

python run_experiment.py --step perturb  

python run_experiment.py --step stats 

python run_experiment.py --step visualize 

python vis_pca_dynamics.py --all_seeds

python vis_plsr_dynamics.py --all_seeds