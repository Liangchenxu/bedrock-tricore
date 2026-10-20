"""Backward-compatible entry point for the V9 verification run.

The implementation lives in ``run_v9_verification.py``. This alias keeps the
old ``python run_v8_verification.py`` command working.
"""

from run_v9_verification import main


if __name__ == "__main__":
    main()
