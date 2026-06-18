"""MEDCU: the retain-subspace residual-weighted entropy forgetting loss.

Self-contained reimplementation of the method used in the paper (no framework
dependency). The forget loss is a per-token *negative entropy* maximised on
forget response tokens, each weighted by omega in [floor, 1] derived from how far
the token's hidden state lies *outside* a low-rank retain subspace:

    L_forget = (1/N_f) * sum_{i,t} omega_{i,t} * ell_ent(y_t),   ell_ent = -H(p) = sum p log p
    omega_{i,t} = floor + (1-floor) * clamp((e_{i,t}-q_lo)/(q_hi-q_lo), 0, 1)
    e_{i,t}     = (1/d) * || (I - P_S) LN(x^f_{i,t}) ||^2 ,  P_S = V_k V_k^T
    V_k         = top-k right singular vectors of the centered retain response reps
    L_total     = gamma * L_forget + alpha * L_retain   (L_retain = cross-entropy on retain)

Reference-free (no second/reference model). See the paper for the full write-up.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def entropy_per_token(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Negative entropy per (shifted) token: -H(p) = sum_i p_i log p_i, shape [B, T-1].

    Minimising this maximises entropy (flattens the next-token distribution toward
    uniform). `mask` is a float [B, T-1] tensor selecting response tokens.
    """
    shift_logits = logits[:, :-1, :].contiguous().float()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    probs = log_probs.exp()
    neg_entropy = (probs * log_probs).sum(dim=-1)  # = -H(p)
    return neg_entropy * mask


def extract_response_hiddens(hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Gather hidden states at response (non-masked) token positions, shape [N, d]."""
    shift_h = hidden[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels != IGNORE_INDEX
    return shift_h[mask]


@torch.no_grad()
def build_retain_subspace(h_resp: torch.Tensor, rank_k: int, max_tokens: int = 4096):
    """Truncated-SVD retain subspace from centered, layer-normed retain response reps.

    Returns (V_k [d, k], mu [d]). V_k spans the top-k retain directions; mu is the
    retain-batch mean used to center forget tokens in the same geometry.
    """
    x = h_resp.float()
    x = F.layer_norm(x, x.shape[-1:])
    if x.shape[0] > max_tokens:
        idx = torch.randperm(x.shape[0], device=x.device)[:max_tokens]
        x = x[idx]
    mu = x.mean(0, keepdim=True)
    xc = x - mu
    M, d = xc.shape
    k = max(1, min(rank_k, min(M, d) - 1))
    try:
        q = min(k + 5, min(M, d) - 1)
        _, _, V = torch.svd_lowrank(xc, q=q, niter=2)
        V_k = V[:, :k].contiguous()
    except Exception:
        _, _, Vh = torch.linalg.svd(xc, full_matrices=False)
        V_k = Vh[:k].T.contiguous()
    return V_k, mu.squeeze(0)


@torch.no_grad()
def residual_energies(h_resp: torch.Tensor, V_k: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Per-token energy outside the retain subspace: e = (1/d) ||(I - V_kV_k^T) LN(x) - mu||^2."""
    x = h_resp.float()
    x = F.layer_norm(x, x.shape[-1:])
    xc = x - mu.unsqueeze(0)
    d = xc.shape[-1]
    norm_sq = (xc * xc).sum(-1)
    proj_sq = ((xc @ V_k) ** 2).sum(-1)
    return ((norm_sq - proj_sq).clamp_min(0.0)) / d


@torch.no_grad()
def quantile_weights(e: torch.Tensor, q_low_p: float, q_high_p: float, weight_floor: float) -> torch.Tensor:
    """Map residual energies to omega in [floor, 1] via batch-quantile gating."""
    if e.numel() < 2:
        return torch.ones_like(e)
    q_low = torch.quantile(e, q_low_p)
    q_high = torch.quantile(e, q_high_p)
    z = ((e - q_low) / (q_high - q_low + 1e-8)).clamp(0.0, 1.0)
    return weight_floor + (1.0 - weight_floor) * z


def medcu_forget_loss(
    forget_logits: torch.Tensor,
    forget_hidden: torch.Tensor,
    forget_labels: torch.Tensor,
    retain_hidden: torch.Tensor,
    retain_labels: torch.Tensor,
    rank_k: int = 64,
    max_retain_tokens: int = 4096,
    quantile_low: float = 0.1,
    quantile_high: float = 0.9,
    weight_floor: float = 0.1,
):
    """Compute the omega-weighted negative-entropy forget loss (a scalar tensor).

    `forget_hidden` / `retain_hidden` are the hidden states from the chosen layer
    (default penultimate); pass them detached. Returns (loss, info_dict).
    """
    h_retain = extract_response_hiddens(retain_hidden, retain_labels)
    h_forget = extract_response_hiddens(forget_hidden, forget_labels)

    omega_flat, e_forget = None, None
    if h_retain.shape[0] >= max(rank_k + 2, 8) and h_forget.shape[0] > 0:
        V_k, mu_r = build_retain_subspace(h_retain, rank_k, max_retain_tokens)
        e_forget = residual_energies(h_forget, V_k, mu_r)
        omega_flat = quantile_weights(e_forget, quantile_low, quantile_high, weight_floor)

    shift_labels = forget_labels[:, 1:].contiguous()
    mask = shift_labels != IGNORE_INDEX
    entropy_grid = entropy_per_token(forget_logits, mask.float())

    if omega_flat is not None:
        omega_grid = torch.zeros_like(entropy_grid)
        omega_grid[mask] = omega_flat.to(omega_grid.dtype)
    else:
        omega_grid = mask.float()

    n_tok = mask.sum().clamp(min=1).float()
    loss = (omega_grid * entropy_grid).sum() / n_tok

    info = {}
    if omega_flat is not None:
        info = {
            "omega_mean": omega_flat.mean().item(),
            "omega_frac_floor": (omega_flat <= weight_floor + 1e-6).float().mean().item(),
            "e_mean": e_forget.mean().item(),
            "n_forget_tokens": int(e_forget.numel()),
        }
    return loss, info
