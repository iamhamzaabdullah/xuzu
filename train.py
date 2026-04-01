#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════╗
# ║  XUZU v1.0.0  ·  Built by Hamza A  ·  terminalBio          ║
# ║  terminalBio Proprietary — All Rights Reserved              ║
# ╚══════════════════════════════════════════════════════════════╝
"""train.py — XUZU training entrypoint"""
import argparse
from xuzu import XUZU, XUZUConfig, XUZUTrainer, build_dataloaders, NucleotideTokenizer

def main():
    p = argparse.ArgumentParser(description="Train XUZU")
    p.add_argument("--data",       required=True, help="Path to .jsonl training file")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--d_model",    type=int,   default=256)
    p.add_argument("--patience",   type=int,   default=10)
    p.add_argument("--device",     type=str,   default="cpu")
    p.add_argument("--save",       type=str,   default="xuzu_best.pt")
    p.add_argument("--log",        type=str,   default="xuzu_train.log")
    p.add_argument("--val_frac",   type=float, default=0.1)
    args = p.parse_args()

    cfg = XUZUConfig(d_model=args.d_model, lr=args.lr, batch_size=args.batch_size)
    tok = NucleotideTokenizer()
    trn, val = build_dataloaders(args.data, tok,
                                  val_frac=args.val_frac,
                                  batch_size=args.batch_size,
                                  max_seq_len=cfg.max_seq_len,
                                  max_poc_len=cfg.max_poc_len)
    model   = XUZU(cfg)
    trainer = XUZUTrainer(model, device=args.device, log_file=args.log)
    trainer.train(trn, val, epochs=args.epochs, patience=args.patience, save_path=args.save)

if __name__ == "__main__":
    main()
