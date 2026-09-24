"""Training package.

The runnable Qwen adapter, SFT loop, and PPO loop enter in M1/M5. Their absence is
intentional at M0: the repository does not claim a one-click trainer before the
native processor and hidden-state readout have been verified.
"""
