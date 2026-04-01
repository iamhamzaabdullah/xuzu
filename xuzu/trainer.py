"""
xuzu.trainer
============
Production trainer: warmup+cosine LR, gradient clip, early stopping,
checkpoint save/restore, per-epoch logging, mixed-precision (AMP).
"""
from __future__ import annotations
import os, time, math, random, logging
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .model   import XUZU, XUZUConfig
from .metrics import evaluate_batch

logger = logging.getLogger("xuzu.trainer")


def _cosine_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * prog))


class XUZUTrainer:
    def __init__(self, model: XUZU, device: str = "cpu",
                 use_amp: bool = False, log_file: Optional[str] = None) -> None:
        self.model   = model.to(device)
        self.device  = device
        self.use_amp = use_amp and device != "cpu"
        self.cfg     = model.cfg
        self.step    = 0
        self.opt = torch.optim.AdamW(model.parameters(),
                                     lr=self.cfg.lr,
                                     weight_decay=self.cfg.weight_decay,
                                     betas=(0.9, 0.999))
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except Exception:
            self.use_amp = False
            self.scaler  = None

        handlers = [logging.StreamHandler()]
        if log_file:
            handlers.append(logging.FileHandler(log_file))
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s | %(levelname)s | %(message)s",
                            handlers=handlers)

    def _set_lr(self, total_steps: int) -> float:
        lr = self.cfg.lr * _cosine_warmup(self.step, self.cfg.warmup_steps, total_steps)
        for g in self.opt.param_groups: g["lr"] = lr
        return lr

    def _train_step(self, batch: dict) -> float:
        self.model.train()
        dev = self.device
        seq_ids  = batch["seq_ids"].to(dev)
        seq_mask = batch["seq_mask"].to(dev)
        adj      = batch["adj"].to(dev)
        poc_ids  = batch["poc_ids"].to(dev)
        poc_mask = batch["poc_mask"].to(dev)
        t_frac   = random.uniform(0.05, 1.0)

        self.opt.zero_grad(set_to_none=True)

        if self.use_amp:
            # Mixed-precision: forward + loss inside autocast, backward via scaler.
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = self.model(seq_ids, seq_mask, adj, poc_ids, poc_mask, t_frac)
                B, L, V = logits.shape
                loss = F.cross_entropy(logits.reshape(B * L, V),
                                       seq_ids.reshape(B * L),
                                       ignore_index=self.model.tokenizer.pad_id)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            logits = self.model(seq_ids, seq_mask, adj, poc_ids, poc_mask, t_frac)
            B, L, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B * L, V),
                                   seq_ids.reshape(B * L),
                                   ignore_index=self.model.tokenizer.pad_id)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

        return loss.item()

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for batch in loader:
            dev = self.device
            seq_ids  = batch["seq_ids"].to(dev)
            seq_mask = batch["seq_mask"].to(dev)
            adj      = batch["adj"].to(dev)
            poc_ids  = batch["poc_ids"].to(dev)
            poc_mask = batch["poc_mask"].to(dev)
            logits   = self.model(seq_ids, seq_mask, adj, poc_ids, poc_mask, 0.5)
            B, L, V  = logits.shape
            loss     = F.cross_entropy(logits.reshape(B*L, V),
                                       seq_ids.reshape(B*L),
                                       ignore_index=self.model.tokenizer.pad_id)
            total += loss.item(); n += 1
        return total / max(n, 1)

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100, patience: int = 10,
              save_path: str = "xuzu_best.pt",
              eval_every: int = 1) -> dict:
        total_steps  = epochs * len(train_loader)
        best_val     = float("inf")
        patience_ctr = 0
        history      = {"train_loss": [], "val_loss": []}

        logger.info(f"XUZU | {self.model.num_parameters():,} params | "
                    f"device={self.device} | total_steps={total_steps}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            trn_l = 0.0
            for batch in train_loader:
                lr     = self._set_lr(total_steps)
                trn_l += self._train_step(batch)
                self.step += 1
            trn_l /= len(train_loader)
            history["train_loss"].append(trn_l)

            if epoch % eval_every == 0:
                val_l = self._validate(val_loader)
                history["val_loss"].append(val_l)
                logger.info(f"Epoch {epoch:>4d}/{epochs} | "
                            f"train={trn_l:.4f} | val={val_l:.4f} | "
                            f"lr={lr:.2e} | {time.time()-t0:.1f}s")

                if val_l < best_val - 1e-5:
                    best_val = val_l; patience_ctr = 0
                    self.model.save(save_path)
                    logger.info(f"  ✓ Best val={best_val:.4f} saved → {save_path}")
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        logger.info(f"  Early stop at epoch {epoch}")
                        break

        logger.info("Training complete.")
        return history
