"""Test configuration for the shared CPU-only validation environment.

The host exposes an NVIDIA device, but this container's NVML/pynvml probe can
block inside ``torch.cuda.is_available()``.  The test suite is deliberately
small and device-agnostic, so force the CPU path before test modules are
imported.  Production code is unchanged.
"""

import sys

# Block the deprecated pynvml shim before importing torch: on this shared host
# its NVML discovery can block indefinitely.  The suite is device-agnostic.
sys.modules["pynvml"] = None

import torch

torch.cuda.is_available = lambda: False
