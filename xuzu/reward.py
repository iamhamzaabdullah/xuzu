"""
xuzu.reward
===========
Proprietary binding affinity surrogate + GER closed-loop refinement engine.

Authors : Hamza A
Lab     : terminalBio
License : terminalBio Proprietary — All Rights Reserved
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class BindingAffinitySurrogate(nn.Module):
    """
    Predicts log10(Kd / nM) from mean-pooled fused context embeddings.
    Trained on (aptamer, target, Kd) triples from BindingDB / literature.
    Lower predicted Kd = higher binding affinity = higher reward.
    """
    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # context: (B, L, D) or (B, D)
        if context.dim() == 3:
            pooled = context.mean(dim=1)
        else:
            pooled = context
        return self.net(pooled).squeeze(-1)     # (B,)

    def reward(self, context: torch.Tensor) -> torch.Tensor:
        """Higher reward = lower predicted Kd."""
        return -self.forward(context)           # negate log(Kd)


class GERLoop:
    """
    Generation–Evaluation–Refinement closed-loop fine-tuner.

    Uses REINFORCE policy gradient to update the diffusion decoder
    toward aptamers with lower predicted Kd (higher binding affinity).

    Reward signal is z-score normalised within each batch to reduce
    high-variance gradient updates.
    """
    def __init__(
        self,
        model:     "XUZU",
        surrogate: BindingAffinitySurrogate,
        device:    str = "cpu",
        lr:        float = 3e-5,
        entropy_coef: float = 0.01,
    ) -> None:
        self.model     = model.to(device)
        self.surrogate = surrogate.to(device)
        self.device    = device
        self.entropy_c = entropy_coef
        # only fine-tune decoder; encoders stay frozen
        self.opt = torch.optim.Adam(
            list(model.decoder.parameters()), lr=lr)

    @torch.no_grad()
    def _compute_reward(self, context: torch.Tensor) -> torch.Tensor:
        raw = self.surrogate.reward(context)        # (B,)
        return (raw - raw.mean()) / (raw.std() + 1e-8)

    def refine_step(
        self,
        context:  torch.Tensor,   # (B, L, D)
        seq_ids:  torch.Tensor,   # (B, L)
        seq_mask: torch.Tensor,   # (B, L)
        t_frac:   float = 0.5,
    ) -> dict:
        self.model.decoder.train()
        logits    = self.model.decoder(seq_ids, context, t_frac, seq_mask)
        log_probs = F.log_softmax(logits, dim=-1)               # (B,L,V)
        probs     = log_probs.exp()

        B, L, V = probs.shape
        sampled  = torch.multinomial(
            probs.view(-1, V), num_samples=1).view(B, L)
        taken    = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        taken    = (taken * seq_mask.float()).sum(-1) / seq_mask.float().sum(-1).clamp(1)

        reward = self._compute_reward(context.detach())

        # entropy bonus — encourages sequence diversity
        entropy = -(probs * log_probs).sum(-1)                  # (B,L)
        entropy = (entropy * seq_mask.float()).mean()

        pg_loss  = -(taken * reward).mean()
        loss     = pg_loss - self.entropy_c * entropy

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.decoder.parameters(), max_norm=0.5)
        self.opt.step()

        return {"pg_loss": pg_loss.item(),
                "entropy": entropy.item(),
                "mean_reward": reward.mean().item()}
