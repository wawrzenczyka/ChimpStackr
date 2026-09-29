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
- [x] Add automatic Near/Far Depth for 2–4 frames using vendored Depth Anything V2 Small at pinned commit `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` and SHA-256-verified official weights.
- [x] When a landscape similarity fit has too few credible inliers, keep identity framing and report `identity_unreliable_match` instead of attempting an unchecked warp.

## Evidence and benchmark

- [x] Locate and preserve the 6000×4000 original pair `Stacking/DSCF9703.JPG` and `Stacking/DSCF9704.JPG`; reproduce the reported SIFT fallback and inspect fixed full-resolution crops of the new composite.
- [ ] Gather held-out real 2-, 3- and 4-frame landscapes, including several near and several far sources, branches, skies, grass, wind and 16-bit inputs.
- [ ] Save fixed 100% crops and score boundary continuity, halos, ghosting, color, landmark residuals, crop loss, runtime and peak RAM/VRAM per scene.
- [ ] Freeze identical aligned inputs for fusion comparisons and identical fusion methods for alignment comparisons.

## Paper reproduction: alignment

- [x] Test SIFT/ECC and RoMa on the supplied pair: SIFT produced 13 ratio matches and 4 inliers; RoMa yielded very low match confidence and a rejected implausible fit. Identity framing is safer for this pair.
- [ ] Benchmark those alignment options on held-out landscapes and report accuracy where reliable shared content exists.
- [ ] Reproduce EfficientLoFTR (CVPR 2024) for speed and match coverage.
- [ ] Evaluate RoMa v2 and MASt3R only if earlier matchers leave measurable focus/parallax failures; check their model terms.
- [ ] Compare conservative global transforms and confidence-gated local correction using foreground/background residuals and finished images.
- [ ] Promote experimental RoMa alignment as a recommended default only after a held-out quality win and practical Windows packaging.

## Paper reproduction: fusion

- [ ] Reproduce DMANet (AAAI 2025) and PDRFuse (Information Fusion 2026) on aligned pairs and inspect the fruit/branch boundary at 100%.
- [x] Audit official PDRFuse source at pinned commit `6253f821334efca6216a0a15918278e20fa74b58`; its required `selective_scan_cuda_core` extension needs a Windows build and a usable MSVC toolchain is not yet confirmed. Audit StackMFF V2 and DMANet artifacts; neither is ready as a redistributable backend without further rights or runnable-code verification.
- [x] Evaluate a learned depth prior on the supplied pair: Near/Far Depth preserves the fruit and main branches much better than the existing focus-only cuts. Record the remaining bokeh halos where both photos lack clean background pixels.
- [x] Run a second 6000×4000 local pair (`DSCF9811.JPG` and `DSCF9812.JPG`) through the same automatic method and inspect its full-frame result.
- [x] Verify the 2–4-frame depth method on source-order and several-near/several-far synthetic focus stacks.
- [ ] Validate 3–4-frame Near/Far Depth on real captured landscapes; the synthetic check alone is insufficient.
- [ ] Reproduce SD-Fuse (Information Fusion 2026) if boundary defects remain; verify its checkpoint and distribution terms.
- [ ] Reproduce StackMFF V2 (EAAI 2025) as the native several-frame baseline on 2–4 source frames.
- [ ] Evaluate DSAF-Net (Information Fusion 2024) as a joint alignment/fusion comparison, recording its small-image training limitation.
- [ ] Track UHD-MFF's 4K data and code release; do not mark it implemented without runnable artifacts.
- [ ] Verify source-pixel color/16-bit fidelity, Windows runtime, model and weight rights, and automatic fallback for each candidate.
- [ ] Integrate only methods that beat the current landscape methods on held-out real scenes, preserving separate results for multiselect.

## Later evaluation

- [ ] Compare FocusDeep and published native-stack models on long macro bursts after the landscape problem is validated.
