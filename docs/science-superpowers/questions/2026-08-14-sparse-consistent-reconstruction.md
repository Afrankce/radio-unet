# Measurement-Consistent Sparse Radiomap Reconstruction

**Research question:** Under the fixed 6.7 GHz, 0-degree, 560/80/160 scene-disjoint protocol with 819 observed valid pixels per map, does a mask-aware sparse-measurement encoder and measurement-consistent flow matching improve missing-region radiomap reconstruction over an environment-only RadioFlow baseline?

**Background / motivation:** The previous direct-concatenation sparse run was evaluated with an under-converged EMA checkpoint and treated a highly sparse map as an ordinary dense image. The new experiment separates environment information from sparse measurements and enforces exact data consistency at observed pixels.

**Hypotheses:**
- H0: Adding sparse measurements through the proposed representation does not improve missing-region reconstruction over the environment-only baseline.
- H1: The mask-aware encoder improves missing-region reconstruction, and the measurement-consistent FM variant further improves the primary missing-region metric while preserving observed values.

**Population & unit of analysis:** The 800 scenes of the 6.7 GHz dataset, restricted to the common 0-degree beam for each of 8x8, 16x16, and 32x32 arrays. The primary paired unit is one scene evaluated jointly across the three array sizes; each array has fixed train/validation/test scene counts of 560/80/160.

**Key variables (operationalized):**
- Target: normalized valid radiomap pixels on the 256x256 grid; source values in (-300, 0) dB are mapped to [0, 1], while -300 and 1000 sentinels are invalid.
- Environment condition: `[Tx_mask, Height, Beam_map]`.
- Sparse measurement: `sparse_map = observation_mask * target`, with exactly 819 deterministic valid-grid observations per sample.
- Primary outcome: pixel-weighted missing-region dB-RMSE on `valid_mask & ~observation_mask`.
- Secondary outcomes: missing dB-MAE, NMSE, PSNR, SSIM; observed maximum/mean absolute error; overall valid-region metrics.

**What counts as an answer:** A lower missing-region error for the mask-aware and consistency-aware variants, with observed error at numerical zero for the projected variant, supports H1. A result in which the proposed variants do not beat the environment-only baseline, after the EMA and training-step controls are applied, disconfirms H1 for this Lite architecture and sampling density.

**Scope & exclusions:** This is an in-distribution single-beam control experiment, not a strict reproduction of the paper's random-instance multi-configuration Task 2. Cross-frequency transfer, multi-beam training, large models, and literal 5%-of-128x128 sampling are out of scope for this run.

**Open questions for prior-work survey:** Compare direct feature concatenation against mask-aware or partial-convolution encoders; distinguish full-target conditional FM from measurement-consistent inpainting FM; control EMA burn-in and observed-value projection.
