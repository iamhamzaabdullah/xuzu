#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════╗
# ║  XUZU v1.0.0  ·  Built by Hamza A  ·  terminalBio          ║
# ║  terminalBio Proprietary — All Rights Reserved              ║
# ╚══════════════════════════════════════════════════════════════╝
"""design.py — XUZU aptamer design entrypoint"""
import argparse
from xuzu import XUZU, XUZUConfig, evaluate_batch

def main():
    p = argparse.ArgumentParser(description="Design aptamers with XUZU")
    p.add_argument("--pocket",    required=True, help="Binding pocket AA sequence")
    p.add_argument("--template",  default=None,  help="Optional seed sequence")
    p.add_argument("--structure", default=None,  help="Dot-bracket secondary structure")
    p.add_argument("--len",       type=int,   default=40)
    p.add_argument("--n",         type=int,   default=10)
    p.add_argument("--temp",      type=float, default=0.85)
    p.add_argument("--top_k",     type=int,   default=0)
    p.add_argument("--model",     default=None,  help="Path to .pt checkpoint")
    p.add_argument("--device",    default="cpu")
    p.add_argument("--rna",       action="store_true", help="Output RNA (T→U)")
    p.add_argument("--evaluate",  action="store_true", help="Print evaluation metrics")
    args = p.parse_args()

    model = (XUZU.load(args.model, device=args.device) if args.model
             else XUZU(XUZUConfig()).to(args.device))
    print(f"[XUZU] {'Loaded: '+args.model if args.model else 'Demo mode (random weights)'}")

    candidates = model.design(
        pocket_seq=args.pocket, template_seq=args.template,
        dot_bracket=args.structure, seq_len=args.len,
        temperature=args.temp, top_k=args.top_k,
        n_candidates=args.n, as_rna=args.rna)

    print(f"\n{'─'*55}\n XUZU — {len(candidates)} Generated Aptamers\n{'─'*55}")
    for i, apt in enumerate(candidates, 1):
        print(f"  [{i:>3}]  {apt}")

    if args.evaluate:
        m = evaluate_batch(candidates, [args.pocket])
        print(f"\n{'─'*55}\n Evaluation Metrics\n{'─'*55}")
        for k, v in m.items(): print(f"  {k:<22} {v}")

if __name__ == "__main__":
    main()
