"""Temporary local bootstrap for this shared validation host.

NVML discovery in the installed ``pynvml`` shim blocks on this machine.  This
module is imported automatically by Python when the repository root is on the
path, so disable NVML probing and hide CUDA before PyTorch initializes.  It is
removed from the release tree after the CPU validation run.
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.modules["pynvml"] = None
