# Prior-work note: paper-faithful FNO for RadioFlow

## Sources checked

- Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*, arXiv:2010.08895v3: https://arxiv.org/html/2010.08895
- Author repository redirect and current official NeuralOperator project: https://github.com/zongyi-li/fourier_neural_operator
- Official FNO theory and implementation guide: https://neuraloperator.github.io/dev/theory_guide/fno.html
- Current official FNO implementation reference: https://neuraloperator.github.io/dev/_modules/neuralop/models/fno.html

## Established method adopted

The original FNO first lifts a pointwise input function to a hidden channel space, applies four Fourier operator layers, and projects the hidden function to the output. A Fourier layer adds a global spectral convolution to a local linear transform before applying a spatial-domain nonlinearity:

```text
v_(l+1)(x) = sigma(W_l v_l(x) + F^-1(R_l F(v_l))(x)).
```

The original 2D experiments use 12 retained modes per spatial axis and hidden width 32. For real-valued 2D inputs, the reference `rfft2` implementation uses two learned complex frequency tensors corresponding to the positive and negative corners of the first transformed axis. The official documentation also retains coordinate-grid embeddings, linear skip connections, full-precision Fourier blocks, optional domain padding, and four FNO layers as the standard construction.

The RadioFlow adaptation adopts this operator unchanged at block level. It treats `[x_t, Tx mask, height, beam map, t_map]` as the input function and appends two normalized spatial-coordinate channels before the lifting map. The output function is the one-channel Flow Matching velocity.

## Deliberate deviations and why

- Hidden width is 40 instead of the paper's 32. Twelve modes are retained unchanged. Width 40 yields approximately 3.70 million real scalar trainable degrees of freedom, keeping the comparison within 10 percent of the 3.99 million-parameter U-Net Lite model.
- Continuous Flow Matching time is broadcast as one input channel. The FNO paper does not solve a conditional Flow Matching vector field, so a time input is mathematically required; channel concatenation changes neither the Fourier block nor its update equation.
- The released 2D execution order uses GELU and no batch normalization, despite the paper's numerical-experiment prose mentioning ReLU and batch normalization. The released execution graph is used for code-level reproducibility.
- The spectral branch is forced to float32 under AMP because the padded 265x265 FFT shape is not guaranteed to be supported by half-precision cuFFT.

## Known artifacts and controls

- **Non-periodic radio-map boundaries:** FFT assumes periodic structure. The released 2D implementation pads the non-periodic domain and crops it after the Fourier layers. The experiment fixes right/bottom padding at nine pixels for every array.
- **Complex parameter undercounting:** `parameter.numel()` counts a complex element once although it contains two real trainable scalars. The report records both tensor-element count and real scalar degrees of freedom.
- **Architecture-capacity confounding:** Width is fixed before results and selected only to match Lite capacity; no validation- or test-selected width/mode sweep is allowed in the primary experiment.
- **Training and sampling confounding:** Data splits, normalization, seed, optimizer, EMA, effective batch, stopping rule, fixed sampling noise, Euler steps, CFG scale, and evaluator are held constant.
- **Condition-dropout semantics:** The FNO has no separate encoder embedding. Its sample-level CFG dropout therefore zeros the three condition channels while retaining `x_t`, `t`, and coordinates. This is fixed in advance and reported as an architecture-specific implementation difference.
- **Single-seed uncertainty:** One fixed seed per array supports a controlled benchmark reproduction, not a statistical claim over random initializations.

## Prior effect size and relationship to prior work

No cited prior work reports an FNO-versus-RadioFlow effect under this radio-map protocol, so there is no defensible external prior effect size. The design instead uses a predeclared smallest engineering effect of interest: a 0.3 dB decrease in the mean test dB-RMSE across the three arrays, with safeguards against isolated regressions.

This study is an architecture-transfer benchmark. It is not a replication of the PDE results in the FNO paper and does not claim that radio propagation maps satisfy the exact PDE families studied there. It tests whether the same global operator parameterization is useful as a conditional Flow Matching velocity backbone on a fixed radio-map benchmark.

