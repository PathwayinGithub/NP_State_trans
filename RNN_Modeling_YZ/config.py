import os
import torch
from dataclasses import dataclass, field
from typing import List


@dataclass
class TaskConfig:
    """Task configuration"""
    # Time structure
    dt: int = 25                      # Time step (ms)
    init_dur: int = 250               # Init period duration
    delay_dur: int = 800              # Delay period duration
    response_dur: int = 250           # Response period duration

    # Dataset sizes
    n_train_trials: int = 4000
    n_val_trials: int = 800
    n_test_trials: int = 800
    n_ablation_trials: int = 1600

    # Paired trials
    use_paired_trials: bool = True

    @property
    def single_trial_steps(self) -> int:
        return (self.init_dur + self.delay_dur + self.response_dur) // self.dt

    @property
    def trial_steps(self) -> int:
        if self.use_paired_trials:
            return self.single_trial_steps * 2
        return self.single_trial_steps


@dataclass
class ModelConfig:
    """Model configuration"""
    input_dim: int = 3
    hidden_dim: int = 256
    output_dim: int = 2

    tau: float = 100.0
    dt: float = 25

    nonlinearity: str = 'tanh'
    output_nonlinearity: str = 'sigmoid'
    init_spectral_radius: float = 0.9


@dataclass
class TrainConfig:
    """Training configuration"""
    seeds: List[int] = field(default_factory=lambda: list(range(20)))# we train 20 random seeds.
    n_epochs: int = 100
    batch_size: int = 128

    lr: float = 2e-3
    weight_decay: float = 0.
    clip_grad: float = 1.0

    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4

    # Orthogonality constraint
    orth_weight: float = 1.0
    orth_method: str = 'trajectory'
    
    # Attractor stability constraint (maintenance trials, delay period)
    stability_weight: float = 0.3

    # Data loading
    num_workers: int = 4
    pin_memory: bool = True

    # Early stopping
    patience: int = 20
    min_delta: float = 1e-5


@dataclass
class AnalysisConfig:
    """Analysis configuration"""
    n_samples_for_characterization: int = 1600   # each type 400 trials
    n_samples_for_trajectories: int = 1600      # each type 400 trials
    n_pca_components: int = 3
    n_ablation_components: int = 2
    ablation_type: str = 'project_out'


@dataclass
class PathConfig:
    """Path configuration"""
    base_dir: str = './experiments'
    exp_name: str = 'sleep_wake_paired'

    @property
    def exp_dir(self) -> str:
        return os.path.join(self.base_dir, self.exp_name)

    @property
    def data_dir(self) -> str:
        return os.path.join(self.exp_dir, 'data')

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.exp_dir, 'checkpoints')

    @property
    def figure_dir(self) -> str:
        return os.path.join(self.exp_dir, 'figures')

    def setup(self):
        for d in [self.data_dir, self.checkpoint_dir, self.figure_dir]:
            os.makedirs(d, exist_ok=True)

    def get_data_path(self, filename: str) -> str:
        return os.path.join(self.data_dir, filename)

    def get_checkpoint_path(self, seed: int) -> str:
        return os.path.join(self.checkpoint_dir, f'model_seed{seed}.pt')


@dataclass
class ExperimentConfig:
    """Full experiment configuration"""
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    device: str = 'cuda'


def get_default_config(gpu_id: int = 0) -> ExperimentConfig:
    config = ExperimentConfig()
    if torch.cuda.is_available():
        config.device = f'cuda:{gpu_id}'
    else:
        config.device = 'cpu'
    config.paths.setup()
    return config


def print_config(config: ExperimentConfig):
    print("\n" + "=" * 60)
    print("Experiment Configuration")
    print("=" * 60)

    print(f"\n[Task]")
    print(f"  Single trial: {config.task.single_trial_steps} steps")
    print(f"  Paired trials: {config.task.use_paired_trials}")
    if config.task.use_paired_trials:
        print(f"  Total trial length: {config.task.trial_steps} steps")
    print(f"  Training set: {config.task.n_train_trials} trials")

    print(f"\n[Model]")
    print(f"  Hidden dim: {config.model.hidden_dim}")
    print(f"  tau: {config.model.tau}")

    print(f"\n[Training]")
    print(f"  Seeds: {config.train.seeds}")
    print(f"  Epochs: {config.train.n_epochs}")
    print(f"  Batch size: {config.train.batch_size}")
    print(f"  Orth weight: {config.train.orth_weight}")

    print(f"\n[Device]: {config.device}")
    print(f"[Output]: {config.paths.exp_dir}")