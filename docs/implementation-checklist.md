# Landscape stacking implementation checklist

Updated 2026-09-29. Progress reflects code in this repository. See the [paper-led research plan](stacking-research-plan.md) for primary papers and candidate constraints.

## Existing product baseline

- [x] Implement and retain `landscape` and `landscape_blend` with native-precision source pixels and valid-pixel handling.
- [x] Add landscape SIFT similarity alignment, ECC refinement, plausibility checks and fallback.
- [x] Add GUI method multiselect that runs each choice separately and keeps separate results, plus alignment options in GUI/CLI.
- [x] Add Pixi environments, tasks and lock file.
- [x] Remove the experimental `foreground_composite` and `layered_focus` methods and manual mask workflow from this direction.
- [x] Record alignment keypoint, match, inlier, scale and fallback diagnostics in the algorithm object.
- [x] Add automatic Near/Far Cut for two frames and Landscape Regions for 2–4 frames, with pinned PyMaxflow 1.3.2, GUI/CLI choices and separate outputs.
- [x] Verify source-preserving graph-cut selection on synthetic focus regions, several source frames and invalid warp borders.
- [x] Add optional RoMa correspondence alignment with guarded similarity fitting, fallback, `romatch==0.1.2` and pinned checkpoint hashes.
- [x] Add a separate Pixi environment with pinned CUDA 12.4 PyTorch wheels; verify PyTorch detects the NVIDIA GPU and RoMa returns GPU correspondences.

## Evidence and benchmark

- [ ] Obtain the **original** near-fruit/far-trees pair and failed output; preserve full resolution and metadata. The chat previews are insufficient for native-resolution validation.
- [ ] Gather held-out real 2-, 3- and 4-frame landscapes, including several near and several far sources, branches, skies, grass, wind and 16-bit inputs.
- [ ] Save fixed 100% crops and score boundary continuity, halos, ghosting, color, landmark residuals, crop loss, runtime and peak RAM/VRAM per scene.
- [ ] Freeze identical aligned inputs for fusion comparisons and identical fusion methods for alignment comparisons.

## Paper reproduction: alignment

- [ ] Benchmark the integrated RoMa (CVPR 2024) option against SIFT/ECC on the original real pairs; the known-transform smoke test is complete.
- [ ] Reproduce EfficientLoFTR (CVPR 2024) for speed and match coverage.
- [ ] Evaluate RoMa v2 and MASt3R only if earlier matchers leave measurable focus/parallax failures; check their model terms.
- [ ] Compare conservative global transforms and confidence-gated local correction using foreground/background residuals and finished images.
- [ ] Promote experimental RoMa alignment as a recommended default only after a held-out quality win and practical Windows packaging.

## Paper reproduction: fusion

- [ ] Reproduce DMANet (AAAI 2025) and PDRFuse (Information Fusion 2026) on aligned pairs and inspect the fruit/branch boundary at 100%.
- [ ] Reproduce SD-Fuse (Information Fusion 2026) if boundary defects remain; verify its checkpoint and distribution terms.
- [ ] Reproduce StackMFF V2 (EAAI 2025) as the native several-frame baseline on 2–4 source frames.
- [ ] Evaluate DSAF-Net (Information Fusion 2024) as a joint alignment/fusion comparison, recording its small-image training limitation.
- [ ] Track UHD-MFF's 4K data and code release; do not mark it implemented without runnable artifacts.
- [ ] Verify source-pixel color/16-bit fidelity, Windows runtime, model and weight rights, and automatic fallback for each candidate.
- [ ] Integrate only methods that beat the current landscape methods on held-out real scenes, preserving separate results for multiselect.

## Later evaluation

- [ ] Compare FocusDeep and published native-stack models on long macro bursts after the landscape problem is validated.
