# Paper-Faithful FNO for RadioFlow Single-Beam Flow Matching

**Research question:** Under the fixed 6.7 GHz, 0 degree, scene-disjoint 560/80/160 protocol, does replacing the RadioFlow Lite U-Net velocity backbone with a paper-faithful two-dimensional Fourier Neural Operator reduce test dB-RMSE across the 8x8, 16x16, and 32x32 arrays while leaving the Flow Matching path, data, optimization budget, and evaluator unchanged?

**Background / motivation:** The existing RadioFlow Lite backbone is a condition encoder plus U-Net-style velocity decoder with cross-attention. The proposed experiment isolates the backbone choice by replacing that network with the `P -> four Fourier layers -> Q` architecture from Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations* (arXiv:2010.08895), adapted only where conditional Flow Matching mathematically requires an interpolated state and a continuous time input. The result informs whether global spectral mixing is a useful alternative to the current multiscale convolutional backbone for dense radio-map generation.

**Hypotheses:**
- H0 (null): The FNO backbone does not reduce the mean of the three array-specific test dB-RMSE values by at least 0.3 dB relative to the frozen U-Net Lite reference, or it improves fewer than two arrays, or it degrades any one array by more than 0.5 dB.
- H1 (directional): The FNO backbone reduces the mean of the three array-specific test dB-RMSE values by at least 0.3 dB, improves at least two of the three arrays, and degrades no array by more than 0.5 dB.

**Population & unit of analysis:** The population is the fixed MultiConfigRadiomap 6.7 GHz, zero-degree single-beam benchmark for the 8x8, 16x16, and 32x32 arrays. Each array uses 560 training scenes, 80 validation scenes, and 160 held-out test scenes from `scene_split_seed42`. A model run is performed independently for each array. Metrics are computed over valid radio-map pixels in the fixed test scenes using the existing evaluator.

**Key variables (operationalized):**
- Primary outcome: test dB-RMSE produced by the existing same-frequency evaluator with the frozen valid-mask and dB inverse-normalization definitions.
- Secondary outcomes: test dB-MAE, NMSE, PSNR, and SSIM from the same evaluator; parameter count, peak GPU memory, training time, and inference time are descriptive efficiency measures.
- Predictor / intervention: velocity-network backbone, comparing the existing locked U-Net Lite model with the paper-faithful FNO2d model.
- Fixed controls: 6.7 GHz; zero-degree beam; per-manifest beam identity; 256x256 resolution; condition `[Tx mask, height, beam map]`; target and valid-mask definitions; seed 42; AdamW; learning rate 1e-3; weight decay 1e-5; 10 percent warmup; EMA 0.999; effective batch size 56; maximum 1000 epochs; patience 20; fixed hash noise; two-step Euler; CFG scale 1.0.

**What counts as an answer:** The experiment gives a confirmatory engineering answer by applying the H1 decision rule to the three fixed test dB-RMSE values. Secondary metrics explain trade-offs but cannot override the primary rule. With one predetermined seed per array, the result is a controlled benchmark comparison rather than a population-level statistical significance claim.

**Scope & exclusions:** This investigation excludes common8 multi-beam prediction, cross-frequency transfer, sparse Task 2/F3 reconstruction, U-FNO, Fourier-attention hybrids, mode/width sweeps selected using test results, CFG sweeps, and claims of statistical significance across random initializations. Any follow-up hyperparameter search is exploratory and must use validation data only.

**Open questions for prior-work survey:** Confirm the original two-dimensional spectral-mode truncation, the non-periodic-domain padding convention, the released-code activation/normalization choice, the correct accounting of complex-valued trainable degrees of freedom, and the minimal way to expose continuous Flow Matching time without changing the FNO block itself.

