"""Muon optimizer and joint Muon+Adam builder for Experiment 2.

Muon (MomentUm Orthogonalized by Newton-schulz) runs standard SGD-momentum
then replaces each 2D parameter's update with the nearest orthogonal matrix.
Only suitable for 2D weight matrices; embeddings, LM head, and scalar parameters
should use Adam/AdamW instead.
"""

import torch
import torch.nn as nn


@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    Uses a quintic iteration whose coefficients maximize the slope at zero.
    Produces something like US'V^T where S' ~ Uniform(0.5, 1.5).
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz.

    Internally runs standard SGD-momentum, then orthogonalizes each 2D
    parameter's update via Newton-Schulz iteration in bfloat16.

    Important: only use for 2D weight matrices (not embeddings, heads, or scalars).
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)
        assert all(isinstance(p, torch.Tensor) for p in params)
        super().__init__([{"params": params}], defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                g = p.grad
                assert g is not None, "Muon requires all parameters to have gradients"

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - momentum)
                if nesterov:
                    g = g.lerp_(buf, momentum)
                else:
                    g = buf

                g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                scale = max(1, p.size(0) / p.size(1)) ** 0.5
                p.data.add_(g, alpha=-lr * scale)


def build_optimizer(
    model: nn.Module,
    *,
    muon_lr: float = 0.01,
    adam_lr: float = 1e-4,
    weight_decay: float = 0.01,
    momentum: float = 0.95,
) -> list[torch.optim.Optimizer]:
    """Build joint Muon+Adam optimizers for a transformer model.

    - Muon: 2D weight matrices (transformer layer weights)
    - AdamW: embeddings, LM head, biases, layer norms, scalar params

    Returns a list of optimizers to step together.
    """
    muon_params = []
    adam_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Muon only for 2D weight matrices that are not embeddings or LM head
        is_2d = param.ndim == 2
        is_embedding = "emb" in name or "embed" in name
        is_lm_head = "lm_head" in name
        is_norm = "ln" in name or "norm" in name
        is_bias = name.endswith(".bias")

        if is_2d and not is_embedding and not is_lm_head and not is_norm and not is_bias:
            muon_params.append(param)
        else:
            adam_params.append(param)

    optimizers = []

    if muon_params:
        optimizers.append(Muon(muon_params, lr=muon_lr, momentum=momentum))

    if adam_params:
        optimizers.append(torch.optim.AdamW(adam_params, lr=adam_lr, weight_decay=weight_decay))

    return optimizers
