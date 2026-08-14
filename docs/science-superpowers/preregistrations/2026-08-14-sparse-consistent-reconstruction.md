# Pre-registration: Measurement-Consistent Sparse Radiomap Reconstruction

**Frozen at commit:** to be stamped by `prereg.sh freeze`
**Question doc:** `docs/science-superpowers/questions/2026-08-14-sparse-consistent-reconstruction.md`
**Analysis plan:** `docs/science-superpowers/plans/2026-08-14-sparse-consistent-reconstruction.md`

## Hypotheses

- H0: The registered D arm does not improve missing-region reconstruction over A.
- H1: D has lower missing-region dB-RMSE than A and exactly preserves observed measurements under projected sampling.

## Primary analysis (exact)

- Compare arm D `multiscale_consistent` against arm A `environment_only`.
- Use the fixed test split of 160 scenes for each of 8x8, 16x16, and 32x32, with the same 819-point masks within each array and the same deterministic initial noise.
- Compute pixel-weighted dB-RMSE after restoring normalized values to dB using `300 * normalized_error`.
- Primary aggregation concatenates missing-valid pixels across all three arrays and all 160 test scenes per array.
- For uncertainty, compute 10,000 paired scene bootstrap replicates with seed 42. Each replicate samples the common scene IDs with replacement and includes all three arrays for every sampled scene.
- No test-based CFG selection, early stopping, mask regeneration, or post-hoc clipping choice is permitted. Prediction clipping is fixed to `[0,1]` before metric calculation for every arm.

## Secondary comparisons

- B versus A: effect of adding direct sparse-map concatenation.
- C versus B: effect of replacing direct concatenation with mask-aware multi-scale encoding.
- D versus C: effect of pinned-observation FM and Euler projection.
- Report missing dB-MAE, NMSE, PSNR, SSIM, overall valid-region metrics, observed mean/max absolute error, parameter count, and validation-selected epoch.

## Prediction and decision rule

- Directional prediction: D-A missing dB-RMSE < 0.
- Support H1 if the pooled point estimate is below zero and its paired bootstrap 95% percentile interval is entirely below zero.
- D observed mean and maximum absolute errors must both be <= `1e-5` normalized units; failure disconfirms the consistency part of H1 even if missing dB-RMSE improves.
- If the interval includes zero or D is worse than A, report the primary result as null for this Lite/sampling protocol. Any alternative sampler, EMA value, or mask density is exploratory.

## Sample size and stopping

- Fixed test sample: 160 scenes x 3 arrays; training and validation are fixed at 560/80 scenes per array.
- Training: maximum 120 epochs, no early stopping before 1000 optimizer steps, then fixed patience 20 on validation missing dB-RMSE.
- No optional stopping, test peeking, or extension of the test sample.

## Multiplicity

- One confirmatory pooled D-vs-A comparison.
- B-vs-A, C-vs-B, D-vs-C, per-array results, CFG alternatives, raw-weight evaluations, and visual case studies are secondary or exploratory and are not used to replace the primary decision.

## Planned deviations

Any change to mask count, split, objective, EMA schedule, sampling steps, or checkpoint selection after this freeze is a protocol deviation. The affected result will be labeled exploratory and the registration will not be edited.

## Frozen data checksums

- /e/datasets/MultiConfigRadiomap/manifests/scene_split_seed42.json 53e9a5015f959665098cfba95cd8e0e0aa612412
