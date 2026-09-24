# Environment baseline — 2026-09-24

Environment name and location: `vision-jev`, `/home/sol/miniconda3/envs/vision-jev`.

| Component | Installed |
|---|---:|
| Python | 3.11.16 |
| PyTorch | 2.14.0+cu130 |
| CUDA build carried by PyTorch | 13.0 |
| Transformers | 5.17.0 |
| PEFT | 0.21.0 |
| Accelerate | 1.15.0 |
| Gymnasium | 1.3.0 |
| MiniGrid | 3.1.0 |
| MiniWoB++ | 1.1.0, editable official source at commit `33c3b4d` |
| Selenium | 4.49.0 |
| Chrome for Testing / ChromeDriver | 154.0.8037.57, Selenium-managed |
| AndroidEnv | 1.3.0 |
| WebLINX | installed from data extra |

Validation: 13/13 tests passed; Ruff check and format passed; mypy strict passed for 17 source files; `pip check` found no broken requirements; repository check passed.

MiniWoB++ official source is pinned at `/mnt/sda1/sol_data/vision-jev/environments/miniwob-plusplus`. A headless `click-test-2-v1` reset returned a 160×210 RGB screenshot and visible DOM candidates. Selenium downloaded its matching browser and driver under `/home/sol/.cache/selenium` (about 416MB); the environment is runnable without a system-wide browser install.

## GPU verification

Host-level verification passed after running outside the restricted development sandbox:

- NVIDIA driver: 580.173.02; driver CUDA capability: 13.0.
- PyTorch reports `torch.cuda.is_available() == true` and 2 CUDA devices.
- Both devices identify as NVIDIA GeForce RTX 4090.
- The driver and PyTorch each report 50,864,390,144 bytes per device (49,140 MiB in `nvidia-smi`).
- A 1024 × 1024 BF16 matrix multiplication and device synchronization completed on each GPU.

The first check was incorrectly run inside a restricted sandbox that could not communicate with NVML or expose the host GPU. That sandbox failure was an isolation artifact, not a missing host driver. GPU environment access is now verified; model training throughput, peak memory, accuracy and multi-process NCCL remain unmeasured until their dedicated runs.

The environment was created from conda-forge because this host had not accepted the Anaconda default-channel Terms of Service. `environment.yml` uses `nodefaults`; Python packages, including PyTorch, are installed from pip extras.
