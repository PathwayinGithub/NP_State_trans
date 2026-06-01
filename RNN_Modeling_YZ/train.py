import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import os

from config import ExperimentConfig
from model import ContinuousTimeRNN, create_model, save_model, load_model
from task import load_datasets, create_dataloaders


# ============= Early Stopping =============
class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss: float, epoch: int) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        return self.early_stop

    def reset(self):
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0


# ============= Loss Functions =============
def compute_task_loss(outputs: torch.Tensor,
                      targets: torch.Tensor,
                      masks: torch.Tensor,
                      criterion: nn.Module) -> torch.Tensor:
    """Compute masked MSE loss over the response period."""
    loss = criterion(outputs, targets)
    loss = loss.mean(dim=-1)
    masked_loss = (loss * masks).sum() / (masks.sum() + 1e-8)
    return masked_loss


def compute_orthogonality_loss(hidden_states: torch.Tensor,
                                metadata: List[Dict],
                                method: str = 'trajectory',
                                n_components: int = 3,
                                use_relative: bool = True,
                                transition_only: bool = True,
                                init_steps: int = 20) -> torch.Tensor:
    """Encourage orthogonal neural trajectories for N->W vs W->N transitions.

    Args:
        hidden_states: (batch, time, hidden_dim)
        metadata: per-sample metadata list
        method: 'trajectory' or 'subspace'
        n_components: number of PCA components for subspace method
        use_relative: if True, subtract h[0] to get relative trajectories
        transition_only: if True, only use the first init_steps time bins
        init_steps: number of init-period time steps (should match metadata['init_steps'])
    """
    device = hidden_states.device

    n2w_hidden = []
    w2n_hidden = []

    for b, meta in enumerate(metadata):
        trial_type = meta['trial_type']
        if transition_only:
            h = hidden_states[b, :init_steps]
        else:
            h = hidden_states[b]

        if use_relative:
            h = h - h[0:1]

        if trial_type == 'nrem_to_wake':
            n2w_hidden.append(h)
        elif trial_type == 'wake_to_nrem':
            w2n_hidden.append(h)

    if not n2w_hidden or not w2n_hidden:
        return torch.tensor(0.0, device=device)

    n2w = torch.stack(n2w_hidden)
    w2n = torch.stack(w2n_hidden)

    if method == 'subspace':
        return _subspace_orth_loss(n2w, w2n, n_components)
    elif method == 'trajectory':
        return _trajectory_orth_loss(n2w, w2n)
    else:
        raise ValueError(f"Unknown method: {method}")


def _subspace_orth_loss(traj1: torch.Tensor, traj2: torch.Tensor,
                        n_components: int = 3) -> torch.Tensor:
    """Penalise overlap between top-k PCA subspaces of two trajectory sets."""
    flat1 = traj1.reshape(-1, traj1.shape[-1])
    flat2 = traj2.reshape(-1, traj2.shape[-1])
    flat1_centered = flat1 - flat1.mean(dim=0, keepdim=True)
    flat2_centered = flat2 - flat2.mean(dim=0, keepdim=True)
    try:
        _, _, V1 = torch.linalg.svd(flat1_centered, full_matrices=False)
        _, _, V2 = torch.linalg.svd(flat2_centered, full_matrices=False)
    except Exception:
        return torch.tensor(0.0, device=traj1.device)
    n_comp = min(n_components, V1.shape[0], V2.shape[0])
    V1_top = V1[:n_comp]
    V2_top = V2[:n_comp]
    overlap = torch.mm(V1_top, V2_top.T)
    orth_loss = (overlap ** 2).sum() / n_comp
    return orth_loss


def _trajectory_orth_loss(traj1: torch.Tensor, traj2: torch.Tensor) -> torch.Tensor:
    """Penalise cosine similarity between mean trajectories at each time step."""
    mean_traj1 = traj1.mean(dim=0)
    mean_traj2 = traj2.mean(dim=0)
    norm1 = mean_traj1.norm(dim=1, keepdim=True) + 1e-8
    norm2 = mean_traj2.norm(dim=1, keepdim=True) + 1e-8
    mean_traj1_normed = mean_traj1 / norm1
    mean_traj2_normed = mean_traj2 / norm2
    cos_sim = (mean_traj1_normed * mean_traj2_normed).sum(dim=1)
    cos_sim = cos_sim[1:]  # skip t=0 (which is zero after relative subtraction)
    return (cos_sim ** 2).mean()



def compute_stability_loss(model,
                           hidden_states: torch.Tensor,
                           metadata: List[Dict],
                           maintenance_only: bool = True,
                           n_perturb=10,
                           perturb_std=0.1,
                           target_rho = 0.85) -> torch.Tensor:

    device = hidden_states.device

    init_steps  = metadata[0]['init_steps']
    delay_steps = metadata[0]['delay_steps']
    response_start = init_steps + delay_steps - 1
    contract_losses = []

    for b, meta in enumerate(metadata):
        trial_type = meta.get('trial_type', meta.get('first_trial_type', ''))
        if maintenance_only and trial_type not in ('nrem_only', 'wake_only'):
            continue

        h_delay = hidden_states[b, init_steps:, :]    # (delay+response, D)

        # Step-to-step velocity vectors and their norms
        speed = h_delay[1:] - h_delay[:-1]            # (delay+response-1, D)
        # contract_losses.append((speed**2).sum())
        n_vel = speed.shape[0]
        if n_vel > 0:
            ramp = torch.linspace(0.05, 1.0, n_vel, device=device).unsqueeze(-1)
            contract_losses.append((speed**2 * ramp).sum())
        
        T = h_delay.shape[0]
        sample_idx = torch.linspace(0, T - 1, min(6, T), dtype=torch.long, device=device)
        h_sample = h_delay[sample_idx].detach()  # (S, D)

        Fh = model.autonomous_step(h_sample)              # F(h)

        for _ in range(n_perturb):
            eps = torch.randn_like(h_sample) * perturb_std
            Fh_eps = model.autonomous_step(h_sample + eps) # F(h + ε)

            ratio = (Fh_eps - Fh).norm(dim=-1) / (eps.norm(dim=-1) + 1e-8)
            excess = torch.relu(ratio - target_rho)
            contract_losses.append(excess.pow(2).sum())


    # Combine
    total_losses = []
    if contract_losses:
        total_losses.append(torch.stack(contract_losses).mean())
    if not total_losses:
        return torch.tensor(0.0, device=device)

    return sum(total_losses)

# ============= Training =============

def train_one_epoch(model, train_loader, optimizer, criterion, device,
                    clip_grad=1.0,
                    orth_weight=0.0,
                    orth_method='trajectory',
                    stability_weight=0.0,
                    contraction_weight=0.0):
    """Run one training epoch and return averaged metrics."""
    model.train()
    metrics = {'loss': 0, 'trial1_loss': 0, 'trial2_loss': 0,
               'orth_loss': 0, 'stability_loss': 0}
    n_batches = 0

    for batch in train_loader:
        t1_inputs  = batch['trial1']['inputs'].to(device)
        t1_targets = batch['trial1']['targets'].to(device)
        t1_masks   = batch['trial1']['masks'].to(device)
        t2_inputs  = batch['trial2']['inputs'].to(device)
        t2_targets = batch['trial2']['targets'].to(device)
        t2_masks   = batch['trial2']['masks'].to(device)
        metadata   = batch['metadata']

        optimizer.zero_grad()

        outputs1, hidden1 = model(t1_inputs, h0=None, return_hidden=True)
        h_final_1 = hidden1[:, -1, :]
        outputs2, hidden2 = model(t2_inputs, h0=h_final_1, return_hidden=True)

        loss_t1 = compute_task_loss(outputs1, t1_targets, t1_masks, criterion)
        loss_t2 = compute_task_loss(outputs2, t2_targets, t2_masks, criterion)
        task_loss = loss_t1 + loss_t2

        # Orthogonality loss
        orth_loss = torch.tensor(0.0, device=device)
        if orth_weight > 0:
            init_steps = metadata[0]['init_steps']
            orth_loss = compute_orthogonality_loss(
                hidden2, metadata, orth_method, init_steps=init_steps
            )

        # Stability loss — only on maintenance trials, only during delay
        stab_loss = torch.tensor(0.0, device=device)
        if stability_weight > 0:
            # Trial1 is always maintenance (nrem_only / wake_only)
            # so maintenance_only=False is fine — they all qualify
            stab_loss = compute_stability_loss(model,
                hidden1, metadata, maintenance_only=False)


        loss = task_loss + orth_weight * orth_loss + stability_weight * stab_loss
        loss.backward()

        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        metrics['loss'] += loss.item()
        metrics['trial1_loss'] += loss_t1.item()
        metrics['trial2_loss'] += loss_t2.item()
        metrics['orth_loss'] += orth_loss.item()
        metrics['stability_loss'] += stab_loss.item()
        n_batches += 1

    return {k: v / n_batches for k, v in metrics.items()}


@torch.no_grad()
def evaluate(model: ContinuousTimeRNN,
             eval_loader: DataLoader,
             criterion: nn.Module,
             device: str) -> Dict[str, float]:
    """Evaluate the model on a validation/test set."""
    model.eval()
    metrics = {'trial1_loss': 0, 'trial2_loss': 0, 'accuracy': 0}
    total_correct = 0
    total_samples = 0
    n_batches = 0

    for batch in eval_loader:
        t1_inputs = batch['trial1']['inputs'].to(device, non_blocking=True)
        t1_targets = batch['trial1']['targets'].to(device, non_blocking=True)
        t1_masks = batch['trial1']['masks'].to(device, non_blocking=True)

        t2_inputs = batch['trial2']['inputs'].to(device, non_blocking=True)
        t2_targets = batch['trial2']['targets'].to(device, non_blocking=True)
        t2_masks = batch['trial2']['masks'].to(device, non_blocking=True)

        metadata = batch['metadata']

        outputs1, hidden1 = model(t1_inputs, h0=None, return_hidden=True)
        h_final_1 = hidden1[:, -1, :]
        outputs2, _ = model(t2_inputs, h0=h_final_1, return_hidden=False)

        loss_t1 = compute_task_loss(outputs1, t1_targets, t1_masks, criterion)
        loss_t2 = compute_task_loss(outputs2, t2_targets, t2_masks, criterion)

        metrics['trial1_loss'] += loss_t1.item()
        metrics['trial2_loss'] += loss_t2.item()

        batch_size = t2_inputs.shape[0]
        init_steps = metadata[0]['init_steps'] 

        for b in range(batch_size):
            out = outputs2[b, init_steps:, :].cpu().numpy()
            tgt = t2_targets[b, init_steps:, :].cpu().numpy()
            pred = np.argmax(out.mean(axis=0))
            true = np.argmax(tgt.mean(axis=0))
            total_correct += int(pred == true)
            total_samples += 1

        n_batches += 1

    metrics['trial2_loss'] /= n_batches
    metrics['total_loss'] = metrics['trial2_loss']
    metrics['accuracy'] = total_correct / total_samples if total_samples > 0 else 0
    return metrics


def train_single_seed(config: ExperimentConfig,
                      seed: int,
                      verbose: bool = True) -> Tuple[ContinuousTimeRNN, Dict]:
    """Train a single model with the given random seed."""
    device = config.device

    torch.manual_seed(seed)
    np.random.seed(seed)
    if 'cuda' in device:
        torch.cuda.manual_seed(seed)

    bundle = load_datasets(config.paths)
    loaders = create_dataloaders(
        bundle,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        pin_memory=config.train.pin_memory)

    model = create_model(config.model, device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n[Seed {seed}] Parameters: {n_params:,}")
        print(f"  Initial spectral radius: {model.compute_spectral_radius():.3f}")

    optimizer = optim.AdamW(
        model.parameters(), lr=config.train.lr,
        weight_decay=config.train.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=config.train.patience)
    criterion = nn.MSELoss(reduction='none')

    early_stopping_patience = getattr(config.train, 'early_stopping_patience', 30)
    early_stopping_min_delta = getattr(config.train, 'early_stopping_min_delta', 1e-4)
    early_stopper = EarlyStopping(
        patience=early_stopping_patience, min_delta=early_stopping_min_delta)

    history = {
        'train_loss': [], 'train_t1_loss': [], 'train_t2_loss': [], 'train_orth': [],
        'val_loss': [], 'val_t1_loss': [], 'val_t2_loss': [], 'val_accuracy': [],
        'lr': [], 'spectral_radius': []
    }

    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0

    pbar = tqdm(range(config.train.n_epochs), desc=f'Seed {seed}', disable=not verbose)

    for epoch in pbar:
        train_m = train_one_epoch(
            model, loaders['train'], optimizer, criterion, device,
            clip_grad=config.train.clip_grad,
            orth_weight=config.train.orth_weight,
            orth_method=config.train.orth_method,
            stability_weight=config.train.stability_weight)

        val_m = evaluate(model, loaders['val'], criterion, device)
        scheduler.step(val_m['total_loss'])

        history['train_loss'].append(train_m['loss'])
        history['train_t1_loss'].append(train_m['trial1_loss'])
        history['train_t2_loss'].append(train_m['trial2_loss'])
        history['train_orth'].append(train_m['orth_loss'])
        history['val_loss'].append(val_m['total_loss'])
        history['val_t1_loss'].append(val_m['trial1_loss'])
        history['val_t2_loss'].append(val_m['trial2_loss'])
        history['val_accuracy'].append(val_m['accuracy'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['spectral_radius'].append(model.compute_spectral_radius())

        if val_m['total_loss'] < best_val_loss:
            best_val_loss = val_m['total_loss']
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        pbar.set_postfix({
            't1': f"{train_m['trial1_loss']:.4f}",
            't2': f"{train_m['trial2_loss']:.4f}",
            'val': f"{val_m['total_loss']:.4f}",
            'acc': f"{val_m['accuracy']:.1%}",
            'es': f"{early_stopper.counter}/{early_stopping_patience}"
        })

        if early_stopper(val_m['total_loss'], epoch):
            if verbose:
                print(f"\n  Early stopping at epoch {epoch+1}! "
                      f"Best epoch: {best_epoch+1}, Best val_loss: {best_val_loss:.6f}")
            break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    history['best_epoch'] = best_epoch
    history['stopped_epoch'] = epoch
    history['early_stopped'] = early_stopper.early_stop

    if verbose:
        print(f"  Training complete: {epoch+1} epochs, Best epoch: {best_epoch+1}, "
              f"Best val_loss: {best_val_loss:.6f}, Final acc: {history['val_accuracy'][-1]:.1%}")

    save_path = config.paths.get_checkpoint_path(seed)
    save_model(model, save_path, seed,
               history['train_loss'], history['val_loss'],
               {'history': history, 'best_epoch': best_epoch})

    return model, history


def train_multiple_seeds(config: ExperimentConfig,
                         verbose: bool = True) -> Dict[int, Tuple[ContinuousTimeRNN, Dict]]:
    """Train models across all configured seeds."""
    results = {}
    for seed in config.train.seeds:
        print(f"\n{'='*60}")
        print(f"Training Seed {seed}")
        print('='*60)
        model, history = train_single_seed(config, seed, verbose)
        results[seed] = (model, history)
        if 'cuda' in config.device:
            torch.cuda.empty_cache()

    if verbose and results:
        print(f"\n{'='*60}")
        print("Training Summary")
        print('='*60)
        for seed, (_, hist) in results.items():
            final_acc = hist['val_accuracy'][-1]
            best_ep = hist.get('best_epoch', 'N/A')
            stopped_ep = hist.get('stopped_epoch', len(hist['val_loss'])-1)
            early = "yes" if hist.get('early_stopped', False) else "no"
            print(f"  Seed {seed}: acc={final_acc:.1%}, "
                  f"best_ep={best_ep+1}, stopped={stopped_ep+1}, early_stop={early}")

    return results


def load_trained_models(config: ExperimentConfig) -> Dict[int, ContinuousTimeRNN]:
    """Load all trained models from checkpoints."""
    models = {}
    for seed in config.train.seeds:
        path = config.paths.get_checkpoint_path(seed)
        if os.path.exists(path):
            model, info = load_model(path, config.device)
            models[seed] = model
            best_ep = info.get('extra_info', {}).get('best_epoch', 'N/A')
            print(f"  Seed {seed}: loaded (best_epoch={best_ep})")
        else:
            print(f"  Seed {seed}: not found")
    return models