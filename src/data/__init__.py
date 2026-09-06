"""Data auditing, splitting, visualization, and loading utilities.

Heavy optional dependencies such as PyTorch are intentionally not imported at
package import time, so the audit can run before the GPU environment is ready.
"""
