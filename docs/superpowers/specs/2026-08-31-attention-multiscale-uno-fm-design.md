# Attention-Conditioned Multiscale UNO-FM for RadioFlow

## Goal

Build a new dense RadioFlow backbone for the frozen 6.7 GHz, zero-degree,
single-beam, scene-disjoint 560/80/160 protocol. The model keeps the existing
RadioFlow condition encoder, enabled CA/SA implementation, Flow Matching
objective, checkpoint format, evaluator, fixed-noise sampler, CFG=1.0, and
two-step Euler solver. Only the green FM velocity backbone changes from four
serial full-resolution FNO blocks to a U-shaped multiscale neural operator.

The implementation baseline is commit
`44df1ee4e0c40d5600bfe5be62ebbab6cede6431`. Existing U-Net, paper-FNO,
full-resolution Attention-FNO, Hybrid FNO-U, sparse Task 2, checkpoints, and
result directories remain read-only.

## Scientific identity

- Model size: `attention_multiscale_uno_lite`
- Backbone: `attention_conditioned_multiscale_uno2d`
- Arrays: `8x8`, `16x16`, and `32x32`, trained independently
- Frequency: 6.7 GHz
- Steering: 0 degrees; beam ID inferred and validated from each manifest
- Split: 560 train / 80 validation / 160 test scenes, scene-disjoint, seed 42
- Resolution: 256 x 256
- Target: normalized radiomap `Y: [B,1,256,256]`
- Condition: `c=[Tx mask,height,beam map]: [B,3,256,256]`
- Valid mask: loss and metrics only; never a model input

## Inputs and condition pyramid

The raw state entrance is unchanged from the full-resolution Attention-FNO:

```text
x_t:                  [B,1,256,256]
c:                    [B,3,256,256]
g_x,g_y:              [B,1,256,256] each
concat(x_t,c,g_x,g_y) [B,6,256,256]
```

`Conv2d(6,32,kernel_size=1)` lifts this tensor to the first state scale.
The coordinate channels are generated internally over `[0,1]`; `g_x` varies
left-to-right and `g_y` varies top-to-bottom. Time is not replicated as an
input channel.

The unchanged `BasicUNetEncoder` consumes `c` and returns:

```text
e0 [B, 32,256,256]
e1 [B, 32,128,128]
e2 [B, 64, 64, 64]
e3 [B,128, 32, 32]
e4 [B,256, 16, 16]
```

Unlike the old full-resolution Attention-FNO, these embeddings are not all
upsampled and summed. Each embedding is fused only at its native scale. `e0`
through `e3` are injected once in the encoder and once in the decoder; `e4`
is injected at the bottleneck. The model therefore owns nine enabled
`CrossAttention` modules.

## State topology

External state channels follow the multiscale U shape:

```text
resolution: 256 -> 128 -> 64 -> 32 -> 16 -> 32 -> 64 -> 128 -> 256
channels:     32     64   128   256   256   256  128    64     32
```

The encoder has four stages plus a bottleneck. The decoder has four mirrored
stages. Encoder outputs are preserved as skip tensors. Downsampling is
`AvgPool2d(2)` followed by a bias-enabled `1x1` projection. Upsampling is
bilinear interpolation with `align_corners=False`, concatenation with the
same-scale encoder skip, and a bias-enabled `1x1` compression. No state-path
`3x3` convolution or transposed convolution is introduced.

## One FNO layer per scale

Each of the nine stages has external state channels `C_i`, condition channels
`E_i`, fixed internal operator width `w=24`, retained modes `m_i`, and padding
`p_i`:

| Stage | Resolution | `C_i` | `E_i` | `m_i` | `p_i` |
|---|---:|---:|---:|---:|---:|
| encoder 0 | 256 | 32 | 32 | 12 | 9 |
| encoder 1 | 128 | 64 | 32 | 12 | 5 |
| encoder 2 | 64 | 128 | 64 | 8 | 3 |
| encoder 3 | 32 | 256 | 128 | 4 | 2 |
| bottleneck | 16 | 256 | 256 | 4 | 1 |
| decoder 3 | 32 | 256 | 128 | 4 | 2 |
| decoder 2 | 64 | 128 | 64 | 8 | 3 |
| decoder 1 | 128 | 64 | 32 | 12 | 5 |
| decoder 0 | 256 | 32 | 32 | 12 | 9 |

The fixed operator width prevents dense spectral weights at 128 or 256
external channels from turning a Lite comparison into a much larger model.
For one stage:

```text
A_i = CrossAttention_i(z_i,e_i)
h_i = Conv1x1(C_i -> 24)(A_i)
h_i = right_bottom_pad(h_i,p_i)
delta_i = SpectralConv2d(24,24,m_i,m_i)(h_i)
        + Conv1x1(24 -> 24)(h_i)
        + Linear(512 -> 24)(swish(t_emb))[:, :, None, None]
delta_i = crop( GELU(delta_i) )
z_i' = A_i + Conv1x1(24 -> C_i)(delta_i)
```

This is one standard spectral operator layer: one spectral branch, one local
pointwise branch, one time branch, and one residual update. The stage does not
contain a second spectral layer, normalization, dropout, factorized spectral
weights, Q/K/V attention, or a learned spatial kernel outside the retained
RadioFlow spatial-attention convolution.

## Exact CA/SA behavior

The existing `CrossAttention` class is retained verbatim. It is not
Transformer cross-attention:

```text
w_c = sigmoid(MLP(GAP(z_i)) + MLP(GMP(z_i)))
w_s = sigmoid(Conv7x7([channel_mean(z_i), channel_max(z_i)]))
A_i = z_i*w_c + z_i*w_s + Conv1x1(e_i)
```

`CrossAttention_old` is never instantiated. Encoder and decoder stages have
separate CA/SA parameters even when they use the same `e_i` tensor.

## Time, CFG, and output

The shared time embedding remains:

```text
t: [B]
sinusoidal_embedding(t,128)
Linear(128,512) -> swish -> Linear(512,512)
```

Every one of the nine stages owns its own `Linear(512,24)` time projection.
The final decoder output is projected by `Conv1x1(32,128)`, GELU, and
`Conv1x1(128,1)` to the FM velocity `v_theta(x_t,t,c)`.

Training-time sample-level CFG dropout remains `0.25`. A dropped sample zeros
both the raw condition used by the lifting path and all five already-computed
condition embeddings, matching the finite-under-AMP semantics of commit
`44df1ee`. CFG scale `1.0` must be exactly equal to the conditional forward.

## Flow Matching and sampling

Training is unchanged:

```text
x_0 ~ Normal(0,I)
t ~ Uniform(0,1), independently per sample
x_t = (1-t)x_0 + tY
u_t = Y - x_0
loss = sum(valid_mask*(v_theta-u_t)^2) / sum(valid_mask)
```

Evaluation uses the EMA `best.pt`, fixed hash noise, CFG=1.0, and two Euler
steps:

```text
x^(1) = x^(0) + 0.5*v_theta(x^(0),0.0,c)
x^(2) = x^(1) + 0.5*v_theta(x^(1),0.5,c)
```

Functional Mean Flow and one-step sampling are excluded from this experiment
so that only the backbone changes.

## Parameter lock

With the modules and bias policy above, the expected counts are:

- Tensor elements: `3,059,355` (each complex element counted once)
- Independent real scalars: `3,925,659` (real and imaginary parts counted)

The real-scalar count is 1.73% below the locked U-Net Lite count of 3,994,859,
so this remains a size-matched Lite comparison. Factory construction must fail
if either count, the external channel tuple, operator width, modes, padding,
number of attention modules, condition features, or CFG dropout changes.

## Code boundary

Create an independent module and experiment identity:

- `model/attention_multiscale_uno.py`
- `training/same_frequency_multiscale_uno_config.py`
- `training/same_frequency_multiscale_uno_trainer.py`
- `run_same_frequency_multiscale_uno.py` with `train`, `select-cfg`, and `test`
  subcommands
- `scripts/run_same_frequency_multiscale_uno_server.sh`

Register the new model in `training/model_factory.py`. Reuse the existing
same-frequency datasets, trainer primitives, evaluator, checkpointing, EMA,
metrics, and sampling code. Do not modify the behavior or hashes of existing
models and configurations. Do not reuse the unfinished `model/hybrid_fno_u.py`
scaffold because it has different input, dropout, attention-injection, skip,
and residual semantics.

Implementation must begin in a new clean worktree created from commit
`44df1ee`; the current `hybrid-fno-u-singlebeam` worktree contains an untracked
test and must remain untouched apart from these design documents.

