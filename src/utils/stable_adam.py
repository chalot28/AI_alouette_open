"""
StableAdamW — Thuật toán tối ưu riêng cho LogAnalyzer.

Tính năng:
  1. Gradient Centralization (GC) — ổn định gradient flow
  2. AMSGrad — ngăn learning rate tăng
  3. Adaptive Warmup — log-spaced learning rate
  4. Parameter-specific weight decay (no decay trên bias/norm)
  5. Gradient clipping mềm — ngăn loss spike

Công thức:
    g_t = ∇L(θ_t)
    g_t = GC(g_t)  # Gradient Centralization
    v_t = β₂·v_{t-1} + (1-β₂)·g_t²
    v̂_t = max(v̂_{t-1}, v_t)  # AMSGrad
    m_t = β₁·m_{t-1} + (1-β₁)·g_t
    m̂_t = m_t / (1 - β₁ᵗ)
    v̂_t = v̂_t / (1 - β₂ᵗ)
    θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε) - η·λ·θ_{t-1}  # weight decay riêng
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Optional

import torch
from torch.optim import Optimizer


class StableAdamW(Optimizer):
    """
    StableAdamW — AdamW ổn định cho log analysis model.

    Args:
        params: Model parameters.
        lr: Learning rate (default: 3e-4).
        betas: Adam betas (default: (0.9, 0.999)).
        eps: Epsilon (default: 1e-8).
        weight_decay: Weight decay (default: 0.01).
        amsgrad: Sử dụng AMSGrad (default: True).
        grad_centralization: Gradient Centralization (default: True).
        clip_norm: Max gradient norm (default: 1.0).
        warmup_steps: Số bước warmup (default: 1000).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = True,
        grad_centralization: bool = True,
        clip_norm: float = 1.0,
        warmup_steps: int = 1000,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta_0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta_1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            grad_centralization=grad_centralization,
            clip_norm=clip_norm,
            warmup_steps=warmup_steps,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", True)
            group.setdefault("grad_centralization", True)
            group.setdefault("clip_norm", 1.0)
            group.setdefault("warmup_steps", 1000)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Thực hiện một optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            amsgrad = group["amsgrad"]
            use_gc = group["grad_centralization"]
            clip_norm = group["clip_norm"]
            warmup_steps = group["warmup_steps"]

            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # ── Gradient Clipping mềm ──
                if clip_norm > 0:
                    grad_norm = grad.norm()
                    if grad_norm > clip_norm:
                        grad = grad * (clip_norm / grad_norm)

                # ── Gradient Centralization ──
                if use_gc and grad.dim() > 1:
                    grad = grad - grad.mean(
                        dim=tuple(range(1, grad.dim())), keepdim=True
                    )

                state = self.state[p]

                # Khởi tạo state
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                # ── Adaptive Warmup (log-spaced) ──
                if warmup_steps > 0 and step <= warmup_steps:
                    # Warmup: từ 0 → lr, theo log scale
                    warmup_factor = math.log(step + 1) / math.log(warmup_steps + 1)
                    current_lr = lr * warmup_factor
                else:
                    current_lr = lr

                # ── Update biased moments ──
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # ── Bias correction ──
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                # ── AMSGrad ──
                if amsgrad:
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                        eps
                    )
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                step_size = current_lr / bias_correction1

                # ── Update ──
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # ── Weight decay riêng ──
                if weight_decay > 0:
                    # Chỉ decay cho weight matrices, không decay bias/norm
                    if p.dim() >= 2:
                        p.add_(p, alpha=-current_lr * weight_decay)

        return loss

    @staticmethod
    def build_for_model(
        model: torch.nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
    ) -> "StableAdamW":
        """
        Tạo optimizer với parameter-specific settings.

        - Weight matrices: weight_decay = 0.01
        - Biases & LayerNorms: weight_decay = 0 (không decay)
        """
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "norm1" in name or "norm2" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return StableAdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
            warmup_steps=warmup_steps,
        )


# ──────────────────────────────────────────────
# LogScheduler — Learning rate scheduler đi kèm
# ──────────────────────────────────────────────


class LogScheduler:
    """
    LogScheduler — LR scheduler log-spaced, kết hợp warmup + cosine decay.

    Công thức:
        lr(t) = base_lr × warmup(t) × decay(t)

    Usage:
        scheduler = LogScheduler(
            optimizer,
            base_lr=3e-4,
            warmup_steps=1000,
            total_steps=50000,
            min_lr_ratio=0.01,
        )
        for epoch in range(num_epochs):
            train()
            scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float = 3e-4,
        warmup_steps: int = 1000,
        total_steps: int = 50000,
        min_lr_ratio: float = 0.01,
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = base_lr * min_lr_ratio
        self.current_step = 0

    def step(self) -> None:
        """Step scheduler."""
        self.current_step += 1
        t = self.current_step

        if t <= self.warmup_steps:
            # Warmup: log scale từ 0 → base_lr
            factor = math.log(t + 1) / math.log(self.warmup_steps + 1)
            lr = self.base_lr * factor
        else:
            # Cosine decay: base_lr → min_lr
            progress = (t - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.base_lr - self.min_lr) * cosine_decay

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        """Lấy learning rate hiện tại."""
        return self.optimizer.param_groups[0]["lr"]
