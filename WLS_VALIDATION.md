# Joint WLS / SID-UNN validation

The Python environment is `D:\文稿\SID-UNN-env`. Downloads, pip caches,
temporary files, Python bytecode, Matplotlib caches, and Torch caches are directed
there by the installation command and `run_sid_joint.ps1`. The environment uses
the already-installed Python 3.12 executable as its base; no new Python download
on C: is needed. Package versions are recorded in `requirements-wls-tested.txt`.

## Validation

`validate_wls.py` executes the original `wls_filter()` and its helper directly
from `utils/dehaze.py`, without importing its unrelated legacy CuPy code.
With alpha = 4 and lambda = 0.35, the maximum absolute differences were:

| Input | Maximum absolute difference |
| --- | ---: |
| 17 x 23 image transmission | 1.83403e-8 |
| 17 x 23 random positive map | 5.93997e-6 |
| Constant map | 3.63876e-13 |

The L1 atmospheric-reconstruction loss produced these raw-parameter gradients:

| Parameter | Autograd | Central finite difference |
| --- | ---: | ---: |
| alpha_raw | 0.00381299460028 | 0.00381299460811 |
| lambda_raw | 0.00306015906834 | 0.00306015907128 |

Network gradients, transmission/guidance gradcheck, independent batch solves,
constant maps, single-pixel/row/column maps, and a 65 x 79 output shape also passed.
The machine-readable results are in `results/wls_validation.json`.

## Implementation

- `SID.py` pads the noise-input dimensions to a multiple of 32, with a minimum
  dimension of 64, and crops the network output to the original image dimensions.
- WLS uses the same log offset as the reference, with float64 solves and an
  explicit relative-residual check. The iteration limit is 2,000: a 1,000-step
  limit proved insufficient for the backward solve on the 128 x 128 experiment.
- Implicit differentiation solves the adjoint WLS system during backward and
  propagates gradients through the edge weights to both physical parameters.
  This avoids retaining thousands of unrolled solver iterations in memory.
- The network remains float32. Alpha and lambda use positive softplus
  parameterizations and are in the optimizer alongside the network weights.

## Run

From this repository in PowerShell:

```powershell
# Validates first; starts the experiment only if every check passes.
.\run_sid_joint.ps1 -Iterations 400 -MaxSize 128 -OutputDir results/sid_joint_128

# Original resolution, default 2,600 iterations (much slower on CPU).
.\run_sid_joint.ps1
```

Outputs include `hazy.png`, `dehazed.png`, `comparison.png` (input on the left),
and `history.json`. Reconstruction PSNR in the training log measures how well
the generated clear image reproduces the hazy input through the scattering
model; it is not PSNR against the clear reference image.
