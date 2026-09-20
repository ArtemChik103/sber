from __future__ import annotations

try:
    from scripts.train_gate import main
except ModuleNotFoundError:  # direct `python scripts/train_constrained_gate.py`
    from train_gate import main


if __name__ == "__main__":
    main()
