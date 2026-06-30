"""Convenience wrapper for the fixed porous-airfoil model runner.

Use this file when you prefer the conventional GitHub entry point:

    python run.py

It delegates to run_porous_models.main(), which reads all settings from
porous_config.py.
"""

from __future__ import annotations

from run_porous_models import main


if __name__ == "__main__":
    main()
