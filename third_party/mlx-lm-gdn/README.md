# MLX-LM mixed-precision gated-delta kernel

The short scalar-gate Metal recurrence in `src/compute/gated_delta.cpp` is adapted from Apple MLX-LM 0.32.2, `mlx_lm/models/gated_delta.py`, `_make_gated_delta_kernel` (unmasked branch).

Source: [MLX-LM 0.32.2 gated_delta.py](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/models/gated_delta.py). The Python package used as the numerical reference is pinned in the root dependency lock. The original MIT permission notice is retained in [LICENSE](LICENSE). Kernel-source copyright: © 2025 Apple Inc.

The C++ adaptation passes BF16 q/k/v and float32 decay/state directly through MLX's C++ custom-kernel interface. It does not import or invoke Python. This is a forward-only interface for D4; target-model backward/LoRA validation remains D5 work.
