import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable
from dataclasses import asdict

from config import ModelConfig


class ContinuousTimeRNN(nn.Module):
    """
    Continuous-time recurrent neural network.

    Dynamics:  tau * dh/dt = -h + f(W_rec @ h + W_in @ x + b)
    Discretised:  h_{t+1} = (1 - alpha) * h_t + alpha * f(W_rec @ h_t + W_in @ x_t + b)
    where alpha = dt / tau
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.tau = config.tau
        self.dt = config.dt
        self.alpha = config.dt / config.tau

        # Network layers
        self.W_in = nn.Linear(config.input_dim, config.hidden_dim, bias=False)
        self.W_rec = nn.Linear(config.hidden_dim, config.hidden_dim, bias=True)
        self.W_out = nn.Linear(config.hidden_dim, config.output_dim, bias=True)

        # Activation functions
        self.activation = self._get_activation(config.nonlinearity)
        self.output_nonlinearity = self._get_activation(config.output_nonlinearity)
        # Initialisation
        self._initialize_weights(config.init_spectral_radius)

    def _get_activation(self, name: str):
        """Return the activation function by name."""
        activations = {
            'tanh': torch.tanh,
            'relu': torch.relu,
            'sigmoid': torch.sigmoid,
            'none': lambda x: x,
            None: lambda x: x,
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name]

    def _initialize_weights(self, spectral_radius: float):
        """Initialise weights with controlled spectral radius."""
        nn.init.xavier_uniform_(self.W_in.weight)

        nn.init.orthogonal_(self.W_rec.weight)
        with torch.no_grad():
            eigvals = torch.linalg.eigvals(self.W_rec.weight)
            current_sr = torch.max(torch.abs(eigvals)).item()
            if current_sr > 0:
                self.W_rec.weight.mul_(spectral_radius / current_sr)
        nn.init.zeros_(self.W_rec.bias)

        nn.init.xavier_uniform_(self.W_out.weight)
        nn.init.zeros_(self.W_out.bias)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialise hidden state to zeros."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self,
                inputs: torch.Tensor,
                h0: Optional[torch.Tensor] = None,
                return_hidden: bool = False,
                ablation_fn: Optional[Callable] = None,
                ablation_direction: Optional[torch.Tensor] = None,
                ablation_start_step: int = 0,
                ablation_end_step: Optional[int] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            inputs: [batch, time, input_dim]
            h0: [batch, hidden_dim] initial hidden state
            return_hidden: whether to return the full hidden-state sequence
            ablation_fn: ablation function to apply
            ablation_direction: [hidden_dim, n_components] ablation direction(s)
            ablation_start_step: time step from which to start applying ablation
            ablation_end_step: time step at which to stop applying ablation (exclusive).
                            None means ablation continues to the end.
                            For optogenetic-style ablation: set to init_steps + delay_steps
                            so that the response period is unperturbed.

        Returns:
            outputs: [batch, time, output_dim]
            hidden_states: [batch, time, hidden_dim] (if return_hidden=True)
        """
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device

        h = h0 if h0 is not None else self.init_hidden(batch_size, device)

        outputs = []
        hidden_states = [] if return_hidden else None

        for t in range(seq_len):
            x_t = inputs[:, t, :]

            # Compute pre-activation
            pre_act = self.W_rec(h) + self.W_in(x_t)

            # Continuous-time dynamics (Euler discretisation)
            h_new = (1 - self.alpha) * h + self.alpha * self.activation(pre_act)

            if (ablation_fn is not None
                    and ablation_direction is not None
                    and t >= ablation_start_step
                    and (ablation_end_step is None or t < ablation_end_step)):
                h_new = ablation_fn(h_new, h, ablation_direction)

            h = h_new

            # Readout
            outputs.append(self.W_out(h))

            if return_hidden:
                hidden_states.append(h)

        outputs = torch.stack(outputs, dim=1)

        if return_hidden:
            hidden_states = torch.stack(hidden_states, dim=1)

        return outputs, hidden_states

    def compute_spectral_radius(self) -> float:
        """Compute the current spectral radius of W_rec."""
        with torch.no_grad():
            W = self.W_rec.weight
            eigvals = torch.linalg.eigvals(W)
            return torch.max(torch.abs(eigvals)).item()
    
    def autonomous_step(self, h: torch.Tensor) -> torch.Tensor:
        """
        One autonomous Euler step with zero input:
            F(h) = (1 - alpha) h + alpha * f(W_rec h + b)

        Used by RNNFixedPointFinder to locate fixed points h* = F(h*).
        """
        pre_act = self.W_rec(h)                       # includes bias
        return (1 - self.alpha) * h + self.alpha * self.activation(pre_act)

    def single_step(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        ablation_fn: Optional[Callable] = None,
        ablation_direction: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pre_act = self.W_rec(h) + self.W_in(x_t)


        h_new = (1 - self.alpha) * h + self.alpha * self.activation(pre_act)

        if ablation_fn is not None and ablation_direction is not None:
            h_new = ablation_fn(h_new, h, ablation_direction)

        output = self.W_out(h_new)
        return output, h_new


def create_model(config: ModelConfig, device: str = 'cpu') -> ContinuousTimeRNN:
    """Create a model instance."""
    model = ContinuousTimeRNN(config)
    return model.to(device)


def save_model(model: ContinuousTimeRNN,
               path: str,
               seed: int,
               train_losses: list,
               val_losses: list,
               extra_info: dict = None):
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': asdict(model.config),
        'seed': seed,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'extra_info': extra_info or {}
    }
    torch.save(checkpoint, path)
    print(f"Model saved: {path}")


def load_model(path: str, device: str = 'cpu') -> Tuple[ContinuousTimeRNN, dict]:
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Reconstruct config
    config = ModelConfig(**checkpoint['config'])
    model = ContinuousTimeRNN(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    info = {
        'seed': checkpoint['seed'],
        'train_losses': checkpoint['train_losses'],
        'val_losses': checkpoint['val_losses'],
        'extra_info': checkpoint.get('extra_info', {})
    }
    print(f"Model loaded: {path}")
    return model, info