# Dex-BlinkMimic engineering design

## Version and applicability

The active fixed-data experiment is described by `configs/dex_blink_recipe.json`
and [the November 7 recipe](deployment/README_recipe_20261107.md). The early
adapter design below remains as historical rationale, not a description of every
current training path.

| Area | Initial / legacy path | Fixed-data recipe (2026-11-07) |
|---|---|---|
| Root correction | Safe config enables root feedback and shaping | Extra root feedback and reward disabled |
| References | Bounded shards, prefetch, difficulty memory and replay invalidation at swaps | Entire compact reference stays resident; no timed swaps or replay clearing |
| Mirror / contact | Initially disabled pending checks | Mirror 0.5 with eligible-window selection; contact weight 0.25 after geometry checks |
| Feature identity | Shard release checks motion/FK and model hashes | Compact receipt checks pack and dataset hashes; it does not lock all feature-code identities |
| Evaluation | Strict and height-diagnostic historical profiles | `jump_tracking_v2`, with global position error reported separately |

Human inputs are converted and assessed in Dex-Retarget.
The local freeze script only copies robot NPZs and preserves historical
quality labels supplied through `--lafan-labels`; it does not solve or regrade
motions. Existing exploratory LAFAN data remain distinct from newly accepted data.

`source_hashes.json` preserves the original 52-file source snapshot. The current
source, configuration, maintenance scripts, tests and documentation are recorded in
[source_hashes_current_20261107.json](source_hashes_current_20261107.json), excluding
backups and generated experiment outputs. This is a file-identity receipt, not a
runtime or motion-quality acceptance report. Regenerate it after changing covered files.

## Historical adapter design

The repository owns its configuration, data contract, simulator model and evaluation history. BlinkMimic supplies the differentiable world model, actor/critic update and optimization mechanisms. Dex supplies joint order, actuator physics, scalable data handling and deployment semantics. Human motion conversion is provided externally by [Dex-Retarget](https://github.com/luckylhf/Dex-Retarget).

## Reuse decisions

- Keep the structured world-model targets and integrator as a pair. Legacy plain-increment world-model states cannot be resumed as structured models.
- Export a dynamic-goal ONNX actor. Preserve complete observation normalization, history layout, action scaling and root feedback identity.
- Keep root correction and root shaping in the first Dex configuration. Root position currently comes from simulator truth.
- Use explicit robot mirror permutations/signs, tracked feet and contact calibration. Geometry, inertial symmetry and contact semantics need separate checks.
- Convert every new human input with Dex-Retarget in its separate environment. Import only completed accepted training exports with matching model/joints/50 Hz and verified receipts and array parity. This package contains no retargeting solver. Feature caches are fingerprinted against the motion and FK implementations, so changed code requires feature regeneration.
- Reuse bounded CPU prefetch, GPU reference capacity, source difficulty memory, replay invalidation and actual coverage reporting.
- Evaluate complete clips independently of short training episodes. Keep strict height/tilt checks and the historical height-diagnostic protocol explicitly identified.
- Evaluate live and EMA independently. Keep dataset identity and evaluation history explicit for every training run.
- A teacher is optional. Establish the first teacher from a qualified checkpoint of this new training run; validate its effect separately.

## Dataset selection and acceptance

Classify bad reference frames independently of policy failures. Include whole-robot collision checks, joint bounds, continuity and sustained floating. Compute IK and feature derivatives on continuous parent takes. Select intervals with context margins and keep original source/frame IDs. Sampling weights refer to parent actions rather than number of derived windows.

Split fixed development probes from a new sealed test set at the parent-action level. GBTC, dance-boy and dance-girl are established development regressions. They have already influenced checkpoint selection.

## Staged implementation

1. Establish a runnable small-motion path, root correction, structured dynamics, mirror tests, ONNX equivalence and native playback.
2. Validate the locked original Blink dependency stack and the Dex model. Validate the actual deployment stack separately. Compiling an MJB in a different MuJoCo version creates a new physical identity requiring replay acceptance.
3. Validate bounded references with resets, then add failure probes whose windows/failure bins carry shard generation and source identity. Invalidate/refill replay on every switch, including device counters; drain side streams before overwriting shared reference storage.
4. Enable CUDA graph collection/update/reset one at a time, check source-switch correctness and numerical parity, then benchmark wall time and update count.
5. Run a paired short test on the same qualified data, exact initial state and seed. Evaluate every 30 minutes, save every 5 minutes, compare common update counts too. Keep teacher, normalization and root shaping identical between arms.
6. Verify a second seed and successive all-regression nodes before long training.

## Interpretation corrections

Dex suffix-based joint names are silently mapped to themselves by the original prefix-only mirror code. Their signs can still flip, so the defect can produce incorrect data without an exception.

A single standing ankle-origin height cannot calibrate all motion contact labels. Use sole collision geometry and validated stance frames. Contact reward is disabled in the initial recipe.

Reference-action division by action_scale does not amplify the physical target when the same scale multiplies the action at deployment. It affects numerical action magnitude and clipping.

The G1 LAFAN1 benchmark has 2.45 hours across 40 clips, so its average clip lasts about 220 seconds. It also exercises long trajectories.

Checkpoint formats, world-model targets, Muon/Adam states, actor buffers, replay generation and physical engine identity are all part of the migration boundary. A single robot adapter file does not cover this work.

## Scope

Use https://github.com/luckylhf/Dex-Retarget for all new human-motion conversion and kinematic quality assessment. BlinkMimic imports accepted outputs without solving or resampling them, and builds its own training features. Existing frozen data and historical evaluations retain their original provenance.
