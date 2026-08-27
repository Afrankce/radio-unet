# Attention-Conditioned Full-Resolution FNO for RadioFlow

## Goal

Run three new dense RadioFlow experiments for the 6.7 GHz, zero-degree,
single-beam, 560/80/160 scene protocol by replacing the green velocity U-Net
with a four-block full-resolution Fourier Neural Operator while retaining the
blue RadioFlow condition encoder and the enabled RadioFlow CA/SA fusion.

The three arrays are 8x8, 16x16, and 32x32. Existing U-Net, paper-FNO, sparse,
and Hybrid FNO-U results and checkpoints are read-only.

## Locked data flow

For `B` samples and `H=W=256`:

- Flow state `x_t`: `[B,1,H,W]`.
- Dense condition `c=[Tx mask,height,beam map]`: `[B,3,H,W]`.
- Normalized coordinates `g_x,g_y`: one channel each, `[B,1,H,W]`.
- Flow time `t`: `[B]`.
- Velocity output `v_theta(x_t,t,c)`: `[B,1,H,W]`.

The valid mask is not a model input. It remains confined to the masked Flow
Matching objective and evaluation metrics.

## Blue condition encoder

The condition encoder is the existing `BasicUNetEncoder` with Lite features
`(32,32,64,128,256,32)`. It returns:

```text
e0 [B, 32, 256,256]
e1 [B, 32, 128,128]
e2 [B, 64,  64, 64]
e3 [B,128,  32, 32]
e4 [B,256,  16, 16]
```

Each `e_i` has an independent bias-enabled 1x1 projection to width `C=40`.
Projected features are bilinearly resized to `(H,W)` with
`align_corners=False` and added without averaging:

```text
e_c = sum_i resize(project_i(e_i), (H,W))
```

Thus `e_c` has shape `[B,40,H,W]`. The blue encoder is the only U-Net-shaped
component. It extracts conditions; it does not process or decode the FM state.

## Pointwise lifting

The raw operator input excludes a replicated time map:

```text
raw = concat(x_t,c,g_x,g_y)  # [B,6,H,W]
z_0 = Conv2d(6,40,kernel_size=1)(raw)
```

The same affine `6 -> 40` map is applied independently at every pixel. It does
not mix neighboring pixels or change spatial resolution.

`g_x` increases from left to right and `g_y` increases from top to bottom,
both over `[0,1]`.

## RadioFlow CA/SA condition fusion

Each of the four FNO blocks owns one existing enabled `CrossAttention(40,40)`
module. This is the repository's real CA/SA implementation, not the dormant
Q/K/V `CrossAttention_old` class:

```text
z_tilde_l = z_l * ChannelAttention(z_l)
          + z_l * SpatialAttention(z_l)
          + Conv2d_1x1_l(e_c)
```

Channel attention uses global average and global maximum pooling. Spatial
attention uses channel mean and channel maximum followed by a 7x7 convolution
and sigmoid.

## Layerwise time conditioning

Time is encoded once into a shared 512-dimensional representation:

```text
h_t = Linear(128,512)(sinusoidal_embedding(t,128))
h_t = swish(h_t)
h_t = Linear(512,512)(h_t)
```

Every FNO block has its own `Linear(512,40)` projection and broadcasts the
result over space. All four blocks receive the same FM time; the block index is
network depth, not an FM integration step.

## Full-resolution FNO backbone

The backbone width is 40, retained modes are `(12,12)`, and there are exactly
four serial blocks. Every block consumes and returns `[B,40,H,W]`; there is no
state downsampling, state upsampling, decoder, or state skip connection.

For block `l=0,1,2,3`:

```text
z_tilde_l = CrossAttention_l(z_l,e_c)
spectral_l = crop(irfft2(R_l * rfft2(pad(z_tilde_l))))
local_l = Conv2d_1x1_l(z_tilde_l)
time_l = Linear_l(512,40)(swish(h_t))[:, :, None, None]
z_(l+1) = GELU(spectral_l + local_l + time_l)
```

The spectral branch applies nine pixels of right/bottom zero padding and crops
back to `(H,W)`. It retains the two Fourier corners implemented by the existing
`SpectralConv2d`. FFT computation remains float32 under an autocast-disabled
guard; the surrounding network may use float16 AMP.

## Pointwise projection

The four-block output is projected per pixel:

```text
v_theta = Conv2d(128,1,1)(GELU(Conv2d(40,128,1)(z_4)))
```

The output is the FM velocity, not a directly regressed radiomap.

## CFG behavior

Training-time sample-level condition dropout remains `0.25`. A dropped sample
zeros the raw three-channel condition before both the condition encoder and the
raw lifting path. This makes unconditional training genuinely condition-free.

At inference, CFG scale `1.0` returns the conditional velocity directly.
Other finite scales use a condition encoded from zeros for the unconditional
branch and the real condition for the conditional branch.

## Flow Matching and sampling

Training is unchanged:

```text
x_0 ~ Normal(0,I)
t ~ Uniform(0,1), independently per sample
x_t = (1-t)x_0 + t*x_1
u_t = x_1 - x_0
loss = valid-mask MSE(v_theta,u_t)
```

Evaluation uses EMA, fixed hash noise, CFG `1.0`, and two-step Euler:

```text
x^(1) = x^(0) + 0.5*v_theta(x^(0),0.0,c)
x^(2) = x^(1) + 0.5*v_theta(x^(1),0.5,c)
```

The complete four-block network is evaluated once per Euler step.

## Experiment protocol

All runs use frequency 6.7 GHz, steering 0 degrees, seed 42, scene-disjoint
560/80/160 splits, learning rate `1e-3`, AdamW weight decay `1e-5`, EMA
`0.999`, maximum 1000 epochs, patience 20, AMP float16, and the existing Lite
micro-batch/accumulation policy. Beam IDs are inferred and validated from each
manifest rather than hard-coded.

Server allocation:

| GPU | Array | Run |
|---:|---|---|
| 0 | 8x8 | independent |
| 1 | 16x16 | independent |
| 2 | 32x32 | independent |

GPU 3 remains unallocated. The new result root is
`/home/wys/radioflow_20260823/results/attention_fno_samefreq_6.7ghz`.

## Verification gates

1. Unit tests prove lifting input order/shape, coordinate direction, five-scale
   aggregation, four block calls, per-layer time influence, enabled CA/SA,
   finite forward/backward, CFG=1 identity, and condition dropout behavior.
2. Factory/config/CLI tests prove an independent checkpoint identity and no
   mutation of the paper-FNO or U-Net registrations.
3. Server preflight validates all three manifests and splits.
4. One optimizer-step CUDA smoke run per array must write a fresh reloadable
   full-state checkpoint before formal launch.
5. Formal launch is accepted only when three live PIDs, three logs, three
   immutable configs, and initial checkpoint/metric activity are observed.

