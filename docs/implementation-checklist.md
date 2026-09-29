# Landscape stacking implementation checklist

Updated: 2026-09-29. Checkboxes reflect work in this repository; research candidates remain described in [the research plan](stacking-research-plan.md).

## Foundation

- [x] Inspect existing stacking, alignment, CLI and Qt result flows.
- [x] Select first milestone: functional 2–4 frame landscape fusion with separate outputs for multiple selected methods.
- [x] Implement `landscape` and `landscape_blend` fusion cores with native-precision source pixels, a confidence map, valid-pixel masking and conservative seams.
- [x] Add both landscape API/config dispatches for aligned and pre-aligned inputs.
- [x] Add synthetic and end-to-end tests for focus masks, color precision, input order, cancellation, CLI and GUI outputs.
- [ ] Add explicit motion/disagreement diagnostics and a mask preview or editing workflow.

## Alignment

- [x] Add a landscape-specific alignment mode with SIFT matches, robust similarity estimation, shared-texture ECC refinement, plausibility checks and a fallback.
- [x] Retain valid-pixel masks and crop shifts for fusion.
- [ ] Expose alignment residuals and feature coverage for inspection.
- [ ] Benchmark current ORB/ECC against SIFT, EfficientLoFTR and RoMa on real focus-bracketed landscapes.
- [ ] Integrate a learned matcher only if it improves finished images and its runtime/weights fit distribution constraints.

## Product integration

- [x] Add Landscape and Landscape Blend to the settings panel and CLI.
- [x] Add method multiselect in the GUI and run selected methods separately, retaining each output for comparison.
- [x] Add landscape alignment choice in the GUI and CLI.
- [x] Make output selection and export use the selected result; preserve original full-precision data.
- [x] Add Pixi runtime/dev environments, tasks and a lock file for Windows.
- [x] Update user documentation and examples.

## Evaluation

- [ ] Build held-out 2-, 3- and 4-frame real landscape benchmark with sharp foregrounds, skies, branches and motion.
- [ ] Reproduce PDRFuse and LightMFF masks at native resolution and compare them with the built-in landscape method.
- [ ] Review 100% crops and measure halos, ghosting, color shifts, alignment residual, crop loss, runtime and memory.
- [ ] Decide whether a learned fusion or alignment model should become an optional packaged backend.

## Later: long bursts

- [ ] Implement and evaluate `regularized_depth` for 20–150 frame macro stacks after the landscape path is validated.
