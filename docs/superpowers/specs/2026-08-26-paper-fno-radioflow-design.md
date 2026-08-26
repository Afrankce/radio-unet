# Paper-Faithful Conditional FNO2d for RadioFlow

## Goal

Replace only the dense RadioFlow velocity network with a two-dimensional Fourier Neural Operator derived from Li et al., while preserving the existing same-frequency Flow Matching data path, optimization controls, checkpoint discipline, sampling procedure, and evaluator.

Primary sources:

- Paper: https://arxiv.org/abs/2010.08895
- Author repository redirect / current official implementation: https://github.com/zongyi-li/fourier_neural_operator

## Fidelity decision

The implementation follows the paper's operator structure and the released PyTorch `FNO2d` execution order:

1. Pointwise lifting map `P`.
2. Four full-resolution Fourier operator blocks.
3. Pointwise projection map `Q`.
4. Two retained Fourier corners for real-valued 2D input.
5. A parallel spatial-domain 1x1 linear map in every Fourier block.
6. GELU after the first three blocks, no activation after the fourth block.
7. Nine pixels of right/bottom padding for a non-periodic domain, followed by cropping.

The paper's numerical-experiment prose mentions ReLU and batch normalization, while the released `FNO2d` code uses GELU and no batch normalization. The released execution graph is chosen because it is concrete, public, and directly reproducible. This choice must be reported explicitly.

## RadioFlow input and output

For a batch size `B` and spatial resolution `H=W=256`:

- Flow state: `x_t` has shape `[B, 1, H, W]`.
- Dense condition: `c=[Tx mask, height, beam map]` has shape `[B, 3, H, W]`.
- Flow time: `t` has shape `[B]` and is broadcast to `t_map` with shape `[B, 1, H, W]`.
- Coordinates: `(g_x,g_y)` are fixed grids in `[0,1]`, each with shape `[B, 1, H, W]`.

The operator input is

```text
a_FM = concat(x_t, Tx mask, height, beam map, t_map)  # 5 channels
z_in = concat(a_FM, g_x, g_y)                         # 7 channels
```

The valid mask is not a model input. It is used only by the existing masked Flow Matching loss and evaluator.

The output has shape `[B,1,H,W]` and represents the conditional velocity

```text
v_theta(x_t, t, c)
```

rather than a directly regressed radio map. This is the only required semantic change from the paper's coefficient-to-solution operator.

## Network dimensions

The fixed primary model uses:

- Fourier layers: 4
- Retained modes: `modes1=12`, `modes2=12`
- Hidden width: 40
- Non-periodic padding: 9 pixels on the right and bottom
- Lifting: `Linear(7,40)` applied pointwise
- Projection: `Linear(40,128) -> GELU -> Linear(128,1)` applied pointwise
- Fourier weights: dense, unfactorized, complex-valued
- Normalization: none
- Dropout inside FNO blocks: none

The paper used width 32 and 12 modes for its 2D problems. Width 40 preserves the paper's 12-mode spectral truncation while matching the existing Lite capacity more closely. Counting each complex scalar as two real trainable degrees of freedom gives approximately 3,698,657 real scalar degrees of freedom, within 10 percent of the locked U-Net Lite count of 3,994,859. The implementation must report both PyTorch tensor-element count and real scalar degrees of freedom so complex parameters are not undercounted.

No width or mode sweep is part of the confirmatory experiment.

## Fourier block

For hidden feature `z_l`:

```text
spectral_l = irfft2(R_l * rfft2(z_l))
local_l    = Conv2d_1x1_l(z_l)
z_(l+1)    = GELU(spectral_l + local_l)  # layers 0, 1, 2
z_4        = spectral_3 + local_3        # final layer
```

For each layer, `R_l` consists of two independently learned complex tensors with shape `[40,40,12,12]`. They multiply:

```text
fft[:, :, :12, :12]
fft[:, :, -12:, :12]
```

All unretained frequencies are set to zero before `irfft2`. The inverse FFT returns to the spatial domain before the local branch is added and GELU is applied. This local nonlinear spatial step is what allows subsequent layers to regenerate high-frequency content even though each spectral branch retains only low modes.

The FFT branch executes in float32 with CUDA autocast disabled because the padded spatial size is 265x265 and half-precision cuFFT does not support every non-power-of-two shape. The surrounding training loop may retain AMP and gradient scaling.

## Continuous Flow Matching

Training retains the existing zero-noise conditional Flow Matching path:

```text
x_0 ~ Normal(0,I)
t   ~ Uniform(0,1)
x_t = (1-t) x_0 + t x_1
u_t = x_1 - x_0
```

The loss remains the valid-mask MSE between `v_theta(x_t,t,c)` and `u_t`. One independent `t` is sampled per sample per optimizer micro-batch; four Fourier layers do not represent four Flow Matching time steps.

Training-time classifier-free condition dropout remains 0.25 at sample level. For FNO, dropping a condition means zeroing all three condition channels for that sample before concatenation; `x_t`, `t_map`, and coordinates remain present. At CFG scale 1.0, inference is exactly the conditional prediction.

Sampling remains fixed two-step Euler:

```text
x^(0) = fixed_hash_noise
t_0 = 0.0
x^(1) = x^(0) + 0.5 * v_theta(x^(0), t_0, c)
t_1 = 0.5
x^(2) = x^(1) + 0.5 * v_theta(x^(1), t_1, c)
```

## Experiment matrix

Three independent runs are launched, one per available server GPU:

| GPU | Array | Frequency | Steering | Split | Seed |
|---|---|---:|---:|---:|---:|
| 1 | 8x8 | 6.7 GHz | 0 degrees | 560/80/160 scenes | 42 |
| 2 | 16x16 | 6.7 GHz | 0 degrees | 560/80/160 scenes | 42 |
| 3 | 32x32 | 6.7 GHz | 0 degrees | 560/80/160 scenes | 42 |

The beam ID is inferred and validated from each manifest rather than hard-coded in the model. Existing records indicate beam IDs 4, 8, and 32 respectively.

## Comparison discipline

The primary comparison uses test dB-RMSE against the frozen U-Net Lite reference for the same array and protocol. dB-MAE, NMSE, PSNR, SSIM, parameter counts, memory, and runtime are secondary or descriptive. Existing U-Net checkpoints and result directories are read-only. FNO runs use new directories and full-state resume checkpoints.

This experiment does not contain cross-attention, spatial attention, U-Net downsampling, skip connections, U-FNO blocks, time FiLM, or validation-selected spectral hyperparameters. Therefore, it tests the complete backbone replacement rather than an attention ablation.

