import numpy as np
import pickle
from typing import Dict, List, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader

from config import TaskConfig, PathConfig


# ============================================================================
# Pairing rules: Trial2 type -> Trial1 type (to establish initial hidden state)
# ============================================================================
PAIRING_RULES = {
    'nrem_to_wake': 'nrem_only',
    'wake_to_nrem': 'wake_only',
    'nrem_only':    'nrem_only',
    'wake_only':    'wake_only',
}

ALL_TRIAL_TYPES = ['nrem_to_wake', 'wake_to_nrem', 'nrem_only', 'wake_only']
STIMULUS_DIRS   = ['left', 'right']


class SleepWakeTask:
    """
    Sleep state switching / maintenance task — Paired Trials (with delay period).

    Trial temporal structure:
      [0, init_steps)                                  : Init period — input signal present, no loss
      [init_steps, init_steps + delay_steps)           : Delay period — zero input, no loss
      [init_steps + delay_steps, total_steps)          : Response period — zero input, loss applied

    Input (3-dimensional, non-zero only during Init period):
      Channel 0 : Target state signal    -1.0 = target NREM / 1.0 = target Wake
      Channel 1 : Current state = NREM    1.0 = currently NREM / 0.0 = otherwise
      Channel 2 : Current state = Wake    1.0 = currently Wake / 0.0 = otherwise

    Output (2-dimensional, target/loss only during Response period):
      Channel 0 : NREM response
      Channel 1 : Wake response
    """

    TRIAL_TYPE_MAP = {
        'nrem_to_wake': ('wake', True),
        'wake_to_nrem': ('nrem', True),
        'nrem_only':    ('nrem', False),
        'wake_only':    ('wake', False),
    }

    CURRENT_STATE = {
        'nrem_to_wake': 'nrem',
        'wake_to_nrem': 'wake',
        'nrem_only':    'nrem',
        'wake_only':    'wake',
    }

    STATE_SIGNAL = {'nrem': -1.0, 'wake': 1.0}

    def __init__(self, config: Optional[TaskConfig] = None):
        if config is None:
            config = TaskConfig()

        self.config = config
        self.dt = config.dt

        self.init_steps     = config.init_dur     // config.dt   # 20
        self.delay_steps    = getattr(config, 'delay_dur', 0) // config.dt  # 32
        self.response_steps = config.response_dur // config.dt   # 20
        self.single_trial_steps = self.init_steps + self.delay_steps + self.response_steps

        self.input_dim  = 3
        self.output_dim = 2
        self.input_noise_std = 0.06

    def _generate_single_trial(self, trial_type: str,
                               is_first_trial: bool = False) -> Dict:
        """Generate a single trial (init + delay + response).

        Note:
            The init direction is fully determined by trial_type
            (via CURRENT_STATE / TRIAL_TYPE_MAP), so no external
           init_dir parameter is needed.
        """
        T = self.single_trial_steps
        inputs  = np.zeros((T, self.input_dim),  dtype=np.float32)
        targets = np.zeros((T, self.output_dim), dtype=np.float32)
        mask    = np.zeros(T, dtype=np.float32)

        target_state, is_switch = self.TRIAL_TYPE_MAP[trial_type]
        current_state           = self.CURRENT_STATE[trial_type]

        # -- Init period (first init_steps bins)
        inputs[:self.init_steps, 0] = self.STATE_SIGNAL[target_state]
        if current_state == 'nrem':
            inputs[:self.init_steps, 1] = 1.0
        if current_state == 'wake':
            inputs[:self.init_steps, 2] = 1.0

        if self.input_noise_std > 0:
            # Add noise only to channels 1 and 2
            noise = np.random.normal(
                0.0, self.input_noise_std,
                size=(self.init_steps, 2)
            ).astype(np.float32)
            inputs[:self.init_steps, 1:3] += noise

        # -- Delay period (init_steps to init_steps + delay_steps)
        # Zero input, zero target, zero mask — already ensured by np.zeros

        # -- Response period (last response_steps bins)
        resp_start = self.init_steps + self.delay_steps
        if target_state == 'nrem':
            targets[resp_start:, 0] = 1.0
        else:
            targets[resp_start:, 1] = 1.0
        mask[resp_start:] = 1.0

        correct_dir = 'left' if target_state  == 'nrem' else 'right'
        current_dir = 'left' if current_state == 'nrem' else 'right'

        return {
            'inputs':           inputs,
            'targets':          targets,
            'mask':             mask,
            'current_rule':     current_state,
            'correct_response': correct_dir,
            'init':         current_dir,
            'is_switch':        is_switch,
            'is_first_trial':   is_first_trial,
        }

    def generate_paired_trial(self, trial_type: str, init_dir: str) -> Dict:
        """Generate a paired trial (Trial1 establishes hidden state, Trial2 is the target task).

        Note:
            The `init_dir` argument is accepted for API compatibility
            but is not used; direction is determined by trial_type.
        """
        first_trial_type = PAIRING_RULES[trial_type]
        trial1 = self._generate_single_trial(
            first_trial_type, is_first_trial=True)

        trial2 = self._generate_single_trial(
            trial_type, is_first_trial=False)

        metadata = {
            'trial_type':         trial_type,
            'first_trial_type':   first_trial_type,
            'init':               trial2['init'],
            'first_init':         trial1['init'],
            'current_rule':       trial2['current_rule'],
            'correct_response':   trial2['correct_response'],
            'is_switch':          trial2['is_switch'],
            'init_steps':         self.init_steps,
            'delay_steps':        self.delay_steps,
            'response_steps':     self.response_steps,
            'single_trial_steps': self.single_trial_steps,
        }

        return {
            'trial1': trial1,
            'trial2': trial2,
            'metadata': metadata,
        }


# ============================================================================
#  Dataset and DataLoader
# ============================================================================

class PairedTrialDataset(Dataset):
    def __init__(self, task: SleepWakeTask,
                 n_trials: int,
                 trial_types: List[str],
                 seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)

        self.task = task
        self.data = []
        n_per_type = n_trials // len(trial_types)

        for trial_type in trial_types:
            for _ in range(n_per_type):
                init_dir = np.random.choice(STIMULUS_DIRS)
                paired = task.generate_paired_trial(trial_type, init_dir)
                self.data.append(paired)

        np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'trial1_inputs':  torch.FloatTensor(item['trial1']['inputs']),
            'trial1_targets': torch.FloatTensor(item['trial1']['targets']),
            'trial1_mask':    torch.FloatTensor(item['trial1']['mask']),
            'trial2_inputs':  torch.FloatTensor(item['trial2']['inputs']),
            'trial2_targets': torch.FloatTensor(item['trial2']['targets']),
            'trial2_mask':    torch.FloatTensor(item['trial2']['mask']),
            'metadata': item['metadata'],
        }


class PreloadedPairedDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'trial1_inputs':  torch.FloatTensor(item['trial1']['inputs']),
            'trial1_targets': torch.FloatTensor(item['trial1']['targets']),
            'trial1_mask':    torch.FloatTensor(item['trial1']['mask']),
            'trial2_inputs':  torch.FloatTensor(item['trial2']['inputs']),
            'trial2_targets': torch.FloatTensor(item['trial2']['targets']),
            'trial2_mask':    torch.FloatTensor(item['trial2']['mask']),
            'metadata': item['metadata'],
        }


def collate_paired_fn(batch: List[Dict]) -> Dict:
    return {
        'trial1': {
            'inputs':  torch.stack([b['trial1_inputs']  for b in batch]),
            'targets': torch.stack([b['trial1_targets'] for b in batch]),
            'masks':   torch.stack([b['trial1_mask']    for b in batch]),
        },
        'trial2': {
            'inputs':  torch.stack([b['trial2_inputs']  for b in batch]),
            'targets': torch.stack([b['trial2_targets'] for b in batch]),
            'masks':   torch.stack([b['trial2_mask']    for b in batch]),
        },
        'metadata': [b['metadata'] for b in batch],
    }


@dataclass
class DatasetBundle:
    train_data:      List[Dict]
    val_data:        List[Dict]
    test_data:       List[Dict]
    ablation_data:   List[Dict]
    task_config:     TaskConfig
    generation_seed: int


def generate_and_save_datasets(config: TaskConfig,
                               path_config: PathConfig,
                               seed: int = 42) -> DatasetBundle:
    print("\n" + "=" * 60)
    print("Generating datasets (Sleep state switching task - Paired Trials with Delay)")
    print("=" * 60)

    np.random.seed(seed)
    task = SleepWakeTask(config)

    print(f"\nTrial structure:")
    print(f"  Single trial : {task.single_trial_steps} bins "
          f"(init={task.init_steps} + delay={task.delay_steps} + resp={task.response_steps})")
    print(f"  Init period  : 0-{task.init_steps - 1} bins"
          f"  (ch0=target state, ch1=current NREM, ch2=current Wake)")
    print(f"  Init input noise std = {task.input_noise_std}")
    print(f"  Delay period : {task.init_steps}-{task.init_steps + task.delay_steps - 1} bins"
          f"  (zero input, no loss)")
    print(f"  Response period : {task.init_steps + task.delay_steps}-{task.single_trial_steps - 1} bins"
          f"  (ch0=NREM response, ch1=Wake response)")
    print(f"\n  Paired Trials: Trial1 ({task.single_trial_steps} bins)"
          f" + Trial2 ({task.single_trial_steps} bins)")
    print(f"    Trial1 establishes initial hidden state (nrem_only / wake_only)")
    print(f"    Trial2 performs the target task (4 types)")
    print(f"    Trial2.h0 = Trial1.h_final")

    print(f"\nGenerating training data: {config.n_train_trials} trials")
    train_dataset = PairedTrialDataset(
        task, n_trials=config.n_train_trials,
        trial_types=ALL_TRIAL_TYPES, seed=seed)

    print(f"Generating validation data: {config.n_val_trials} trials")
    val_dataset = PairedTrialDataset(
        task, n_trials=config.n_val_trials,
        trial_types=ALL_TRIAL_TYPES, seed=seed + 1)

    print(f"Generating test data: {config.n_test_trials} trials")
    test_dataset = PairedTrialDataset(
        task, n_trials=config.n_test_trials,
        trial_types=ALL_TRIAL_TYPES, seed=seed + 2)

    print(f"Generating ablation data: {config.n_ablation_trials} trials")
    ablation_dataset = PairedTrialDataset(
        task, n_trials=config.n_ablation_trials,
        trial_types=ALL_TRIAL_TYPES, seed=seed + 3)

    bundle = DatasetBundle(
        train_data=train_dataset.data,
        val_data=val_dataset.data,
        test_data=test_dataset.data,
        ablation_data=ablation_dataset.data,
        task_config=config,
        generation_seed=seed,
    )

    save_path = path_config.get_data_path('datasets.pkl')
    print(f"\nSaving data to: {save_path}")
    with open(save_path, 'wb') as f:
        pickle.dump(bundle, f)

    print("Data generation complete!")
    return bundle


def load_datasets(path_config: PathConfig) -> DatasetBundle:
    load_path = path_config.get_data_path('datasets.pkl')
    print(f"Loading data: {load_path}")
    with open(load_path, 'rb') as f:
        return pickle.load(f)


def create_dataloaders(bundle: DatasetBundle,
                       batch_size: int = 64,
                       num_workers: int = 4,
                       pin_memory: bool = True) -> Dict[str, DataLoader]:
    use_cuda = torch.cuda.is_available()
    loader_kwargs = {
        'batch_size': batch_size,
        'collate_fn': collate_paired_fn,
        'num_workers': num_workers if use_cuda else 0,
        'pin_memory':  pin_memory  if use_cuda else False,
    }
    loaders = {
        'train': DataLoader(PreloadedPairedDataset(bundle.train_data),
                            shuffle=True,  **loader_kwargs),
        'val':   DataLoader(PreloadedPairedDataset(bundle.val_data),
                            shuffle=False, **loader_kwargs),
        'test':  DataLoader(PreloadedPairedDataset(bundle.test_data),
                            shuffle=False, **loader_kwargs),
        'ablation': DataLoader(PreloadedPairedDataset(bundle.ablation_data),
                               shuffle=False, **loader_kwargs),
    }
    return loaders