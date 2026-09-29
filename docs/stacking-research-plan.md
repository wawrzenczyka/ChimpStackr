# Focus stacking research and implementation plan

Research checked 2026-09-29. Here, *stacking* means fusing a focus bracketed image burst; HDR exposure fusion is a separate task. This is a plan, not a claim that any proposed method has already beaten ChimpStackr's current output.

## Recommendation

Prioritize **two to four frame landscape stacks**. Evaluate [PDRFuse](https://github.com/zwy0913/PDRFuse) first: its coarse-to-fine decision map targets high-resolution image pairs, and its repository provides source, weights and an Apache-2.0 license. Compare it with the smaller [LightMFF](https://github.com/Xinzhe99/LightMFF) model and a ChimpStackr-native, full-resolution soft-mask baseline. Use the learned method to estimate *where* each source is sharp, then composite from ChimpStackr's aligned `float32` source images to preserve color and 16-bit output. Keep the existing Pyramid default until real landscape photographs establish a win. Work on long macro bursts remains a separate second track.

This is a benchmark-led recommendation, not a declaration of a universal winner. [PDRFuse's authors](https://github.com/zwy0913/PDRFuse) report stronger high-resolution fusion than fifteen compared methods; [UHD-MFF/UMF-LUT](https://arxiv.org/abs/2606.31242), accepted by ECCV 2026, reports real-time 4K fusion and introduces 150 real 4K pairs plus 1,800 synthetic pairs. Its public [repository](https://github.com/zyb5/UHD-MFF) currently contains a README and license but no implementation or weights, so it is a research and dataset lead, not an immediately reproducible integration candidate. [Explicit defocus blur modelling (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/32714) is another relevant model for difficult focus boundaries. Reported scores from different datasets and metrics are not directly comparable.

## First track: two to four frame landscapes

The quality target is a natural all-in-focus photograph with stable colors and no doubled edges. The hard cases are a very near foreground against distant scenery, a soft transition through the middle ground, thin branches against the sky, and changes between exposures caused by wind, water or clouds. Skies and smooth surfaces carry little focus evidence; texture alone cannot determine their source. Alignment must be judged in sharply focused shared regions, with focus breathing corrected before fusion.

**Two-frame algorithm design:**

1. Align the near- and far-focused images and retain warp validity masks. Estimate a coarse focus preference from multi-scale sharpness and, for the learned variant, refine it with PDRFuse. Record the continuous foreground weight `w` and confidence; retain a deterministic baseline when a model is unavailable.
2. Refine the mask at native resolution using image edges and local focus evidence. Preserve broad, coherent near/far regions; use a narrow soft transition only where both images have credible focused detail. Protect branches and other thin foreground structures from erosion. Keep low-texture sky assigned consistently to one source.
3. Detect disagreement that focus alone cannot explain: moving foliage, flowing water, clouds, and parallax. Mark these as low-confidence regions and choose one source locally, rather than blending two different moments into a ghost. Provide a mask preview and brush editing for these cases; automatic correction cannot recover missing scene content.
4. Composite the **original aligned BGR `float32` pixels** with the refined mask. Do not use the reference repositories' JPEG output paths: [PDRFuse's inference script](https://github.com/zwy0913/PDRFuse/blob/main/inference.py) uses default `cv2.imread` and writes JPG, and [LightMFF's script](https://github.com/Xinzhe99/LightMFF/blob/main/predict.py) converts the result to `uint8` before JPG. Integrating only the mask inference separates model input normalization from final pixel precision.

**Three or four frames:** implement a direct N-way confidence map and spatially coherent labels as the baseline. Pairwise networks may be evaluated, but do not repeatedly fuse intermediate RGB results. If a pair model wins, derive each frame's confidence against a common reference or from pairwise masks, normalize across all frames, and verify that reversing input order leaves the output effectively unchanged. Treat this extension as unproven until tested; a strong pair benchmark does not establish strong four-frame performance.

**Landscape evaluation:** build a held-out set of real 2-, 3- and 4-frame RAW/TIFF sequences at native resolution, including 20–60 MP files, near/far boundary detail, dark foregrounds, sky, fine foliage, wind and water. Compare PDRFuse, LightMFF, ChimpStackr's Pyramid/Depth Map/Weighted methods and the native mask baseline using identical aligned sources. Measure mask edge placement, halos, ghosting, source color change, runtime and memory; inspect blinded 100% crops. Use [MFIFB](https://arxiv.org/abs/2005.01116) and the [UHD-MFF paper's real/synthetic split](https://arxiv.org/html/2606.31242) as supplementary external checks where data rights allow. [Evaluation-metric research](https://pubmed.ncbi.nlm.nih.gov/38376964/) argues against selecting a winner from a single fusion score.

**Landscape implementation order:** (1) curate and baseline real landscape pairs, including alignment; (2) run PDRFuse and LightMFF in isolated environments and inspect their masks at native image size; (3) build the CPU soft-mask compositor and disagreement detector; (4) compare a few-frame N-way extension; (5) integrate the winning pair and alignment methods in `AlgorithmConfig`, `API.py`, CLI and GUI. Package model weights only after checking their distribution terms and cross-platform CPU/GPU behavior. Expose the mask for retouching.

### Alignment is a quality gate

ChimpStackr currently uses ORB feature matching and RANSAC for a similarity or affine estimate, then multi-scale ECC refinement; translation can use DFT. That is a useful baseline, but it has not been shown to be state of the art for landscape pairs with different focus. Its grayscale matching and ECC inputs are converted to 8-bit, and ECC refinement can consider blurry or moving regions. [Current implementation](../src/algorithms/__init__.py).

**Plan for landscape alignment:**

1. **Measure the current path first.** For each scene, compare no alignment, similarity and affine alignment. Record matched features, spatial coverage, reprojection error, valid output area and a visual overlay. Inspect residual misalignment on stationary structures that are reasonably sharp in *both* frames; exclude sky, moving foliage and heavily defocused regions from this check.
2. **Benchmark modern correspondences.** Evaluate [EfficientLoFTR](https://github.com/zju3dv/efficientloftr) and [RoMa](https://github.com/Parskatt/RoMa) against ORB, plus OpenCV SIFT as a classical reference. EfficientLoFTR is a CVPR 2024 semi-dense matcher; RoMa is a CVPR 2024 dense matcher with match certainty. Their general image-matching results do not establish performance on focus-bracketed landscapes, so use the curated real pairs. [EfficientLoFTR paper](https://arxiv.org/abs/2403.04765), [RoMa paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Edstedt_RoMa_Robust_Dense_Feature_Matching_CVPR_2024_paper.pdf).
3. **Estimate conservative geometry.** Use high-confidence, spatially distributed matches from stationary areas; fit a similarity transform with robust outlier rejection and refine it on shared focused texture. Consider affine or homography only when held-out residuals improve across the frame without implausible shear, perspective, cropping or doubled edges. Avoid treating dense learned correspondences as permission to warp every pixel independently: parallax and subject motion can make that visually destructive.
4. **Check focus-specific alternatives.** [DSAF-Net](https://www.sciencedirect.com/science/article/pii/S1566253523004414) jointly models defocus, optical flow and fusion for misaligned focus sets. Use it as a research comparison for difficult failures; its close-range synthetic training setup does not prove landscape generalization. Test local flow correction only for repeatable residuals on static regions after global registration, with strict confidence and smoothness limits.
5. **Select by final photographic quality.** Warp original full-precision images once using the chosen transform and produce validity masks. Reject low-confidence estimates or fall back to the current method. Compare each matcher/warp combination with the *same* fusion masks; score double edges, halos, crop loss and preserved straight structures, alongside geometric error. A registration that wins a generic matching benchmark but worsens the finished photograph does not ship.

For three or four frames, align each image to one explicit reference view and record reusable transforms. Compare direct-to-reference alignment with neighboring-frame composition where focus overlap is weak, then validate the composed transforms against the reference. Deliver a debug overlay and a user-adjustable/manual alignment fallback for cases with no trustworthy shared detail.

## What exists in this repository

| Area | Current behavior | Consequence for new work |
| --- | --- | --- |
| Fusion | `laplacian` fuses frames pairwise at each pyramid level; `weighted_average` accumulates Laplacian-energy weights; `depth_map` takes the largest multiscale SML score per pixel. | Compare every candidate with all three existing focus methods, especially Pyramid on hairs and Depth Map on opaque surfaces. |
| Depth data | `_depthmap_core` keeps the winning pixel and best score, then sets `self.depth_map = None`. Its bilateral filter smooths each score map using that score map as its guide. | A globally refined index map requires retaining frame indices and reconstructing pixels after map refinement. Scene edges must guide any new spatial refinement. |
| Alignment | ORB+RANSAC estimates similarity/affine geometry, ECC refines it, and DFT can handle translation. Each frame can be registered against the reference and returns a full aligned `float32` BGR frame. | Benchmark current registration against learned correspondences on focus-bracketed landscapes. A second fusion pass must reproduce the chosen transform exactly; cache transforms or aligned frames. |
| Images | `float32` processing represents source values on a 0–255 scale; PNG/TIFF can be saved at 16 bits. | Preserve that precision. Do not route the new fusion path through 8-bit conversion or CLAHE. Test 16-bit gradients and color fidelity. |
| Product | `AlgorithmConfig`, `API.py`, CLI and Qt settings dispatch methods; CPU is required and CuPy is optional. | Introduce a new method key and leave existing keys and defaults stable during evaluation. |

Relevant code: [`cpu.py`](../src/algorithms/stacking_algorithms/cpu.py), [`API.py`](../src/algorithms/API.py), [`config.py`](../src/config.py), [`cli.py`](../src/cli.py), [`SettingsWidget.py`](../src/MainWindow/SettingsWidget.py).

## Research shortlist

| Approach | Evidence and fit | Main constraint | Decision |
| --- | --- | --- | --- |
| PDRFuse (high-resolution pairs) | Progressive coarse-to-fine decision refinement; official repository includes inference code, weights and Apache-2.0 license. [Code and paper information](https://github.com/zwy0913/PDRFuse). | The reference script reads ordinary 8-bit images and writes JPEG; need native-bit-depth source compositing and real landscape validation. | **Primary pair model to reproduce.** |
| LightMFF (pairs) | Very small learned mask model with official inference code, weights and MIT license. Its authors report 0.02M parameters and 0.02 seconds per pair, under their test setup. [Official repository](https://github.com/Xinzhe99/LightMFF). | Published timing does not establish performance at 20–60 MP; reference output is 8-bit JPEG. | Fast pair baseline and optional low-resource candidate. |
| UHD-MFF / UMF-LUT (pairs) | ECCV 2026 paper targets 4K through coarse region and fine edge lookup tables; introduces a real 4K pair dataset. [Paper](https://arxiv.org/abs/2606.31242). | [Public repository](https://github.com/zyb5/UHD-MFF) has no runnable method or weights yet. | Track release; compare if reproducible artifacts appear. |
| Explicit defocus blur model (pairs) | AAAI 2025 method explicitly estimates spatially varying defocus and special-cases uncertain boundaries. [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/32714). | Integration complexity and reproducibility need checking. | Research comparison for hard boundaries. |
| Native soft-mask / regularized focus selection | Uses multiscale focus evidence and coherent decisions; retains source colors with current NumPy/OpenCV/SciPy/Numba stack. This is a proposed ChimpStackr design, **not a published SOTA result**. [FocusDeep discussion of classical methods](https://arxiv.org/html/2311.17846). | Occlusion and defocus spread still leave some background detail unavailable. | Implement for pair fallback and later long bursts. |
| FocusDeep (2023) | Uses 94 real high-resolution 30-frame RAW bursts, with 84 train and 10 test bursts; reported test PSNR/SSIM against Helicon Focus pseudo ground truth. Its RGB model reported 38.47 dB / 0.965 on that test set, which measures agreement with the teacher, not absolute photographic truth. [Paper](https://arxiv.org/html/2311.17846), [code/data](https://github.com/araujoalexandre/FocusStackingDataset). | Fixed 30-frame architecture, requires good alignment, training targets inherit commercial stack errors. | Benchmark as a realistic neural reference. |
| StackMFF V1/V2/V3 | Whole-stack networks are more directly relevant than pairwise transformer fusion. V2 predicts focal depth; V3 uses pixelwise classification and ships inference scripts. [V2 paper](https://www.sciencedirect.com/science/article/pii/S0952197625026983), [V3 repository](https://github.com/Xinzhe99/StackMFF-V3). | Full stack inference, GPU memory, input bit depth, real macro generalization and model licensing must be checked. V3 publication was in review in the authors' series listing. | V3 is the primary neural spike; include V2 as a fallback comparison. |
| Pairwise transformer fusion, e.g. MFFT or SwinMFF | Active research on image-pair fusion. [MFFT paper/code](https://www.sciencedirect.com/science/article/pii/S0952197624001258), [SwinMFF code](https://github.com/Xinzhe99/SwinMFF). | High-resolution deployment and source-color fidelity require measurement; repeated pair fusion can be order-dependent. | Compare on landscape pairs if PDRFuse and LightMFF leave clear failures. |
| GMFF / StackMFF V4 with diffusion restoration | Attempts to repair uncertain boundaries and missing focal planes. [Preprint](https://arxiv.org/abs/2512.21495). | Generated content may not correspond to any photographed detail; extra runtime and model distribution burden. | Research only; any future feature must be explicitly labeled as restoration. |

## Second track: long macro bursts

The earlier 100–150-frame proposal remains useful for macro and microscopy stacks. It follows the landscape work because its input geometry, memory use and depth-boundary behavior are different. FocusDeep trained on 30-frame real RAW bursts and reported good quality and noise tolerance, but its paper identifies fixed burst length, alignment dependence and memory limits for hundreds of frames. StackMFF V2 regresses focal depth and V3 predicts a per-pixel source class; V3 and V4 were listed as in review by their authors. [FocusDeep paper](https://arxiv.org/html/2311.17846), [StackMFF series](https://github.com/Xinzhe99/StackMFF-Series).

### Proposed deterministic method

Call the prototype `regularized_depth` so the current `depth_map` output remains reproducible.

1. **Register and validate.** Preserve numerical frame order. Record each source-to-reference warp, its valid-pixel mask, and any alignment failure. Exclude invalid borders from focus scoring. If transforms cannot be replayed, cache aligned frames in a temporary store and clean it on cancel or failure. Never use a warped black border as a candidate.
2. **Score focus per frame.** Start with the existing modified Laplacian at multiple scales. Normalize for local luminance and estimate a noise floor so sensor noise and clipped highlights do not win solely by contrast. Compare a noise-aware score with the existing SML score as an ablation; do not assume the new score is superior. For an ordered burst, use a short window along the focus axis to suppress isolated score spikes while preserving a sharp peak.
3. **Keep candidates and confidence.** Stream through frames and retain the top three or four scores and their `uint16` frame indices per pixel. Define confidence from the best-to-runner-up margin divided by their combined score plus an epsilon. Support more than 255 frames and reject stacks beyond the chosen index type's capacity. Preserve the original winning index for comparisons.
4. **Regularize only uncertain labels.** Keep the provisional winning image during the first pass and use it as the full-resolution scene guide for an edge-aware local vote among retained candidate labels. Weight neighbors by spatial distance, color similarity and candidate score. Keep high-confidence labels fixed. Limit voting across strong edges and use a small iteration cap. Inspect thin hairs separately: a smooth depth surface is inappropriate when foreground and background overlap. Keep a fallback to the unregularized winner wherever the vote is ambiguous.
5. **Reconstruct and blend conservatively.** Replay aligned sources to gather the selected pixels. Blend only within a narrow seam where both neighboring candidates are sharp and geometrically valid; otherwise use the selected source. Keep BGR `float32` through output and expose the source-index and confidence maps for diagnostics and retouching. Cancel and pause checks must cover both passes.

The top-four index and score arrays cost roughly 480 MB for a 20-megapixel image (`4 × (2 + 4) × 20M`), before the output, guide, masks and alignment buffers. A naive full focus volume for 150 such frames costs about 12 GB just for `float32` scores. Use a configurable memory budget, bounded frame prefetch and a tiled fallback with a halo wider than the focus and voting kernels. Document peak resident memory and temporary disk use for each mode. Re-reading aligned sources for reconstruction is slower but avoids holding all 150 RGB frames in RAM.

### Long-burst implementation sequence

#### 0. Establish evidence and baseline

- Gather permissioned, representative stacks: at least 20 scenes spanning 2–4, 20–40 and 100–150 frames; macro hairs, opaque textured subjects, smooth surfaces, highlights, noise, focus breathing, alignment failure and 16-bit RAW/TIFF. Keep capture order and camera details. Exclude benchmark scenes from any neural fine tuning.
- Save identical alignment and output settings for Pyramid, Weighted and Depth Map. Record runtime, peak RSS/VRAM, temporary disk and failure rate. Produce fixed crops for blind review. Existing CLI `--quality-report` sharpness is only a diagnostic, not a fusion quality score.
- Use the [LSFD real-burst dataset](https://arxiv.org/html/2311.17846) and a public pair benchmark such as [MFIFB](https://arxiv.org/abs/2005.01116) as supplementary checks. Public pair results must not decide long-burst quality. The 2024 [metric study](https://pubmed.ncbi.nlm.nih.gov/38376964/) found that popular fusion metrics disagree; pair PSNR/SSIM with reference images and blinded visual review.

#### 1. Prototype `regularized_depth` on CPU

- Add pure fusion functions for scoring, top-K update, confidence, label refinement and pixel gather in a new module under `src/algorithms/stacking_algorithms/`. Keep them independent of Qt.
- Add the replayable alignment interface in `src/algorithms/__init__.py` and a two-pass orchestrator in `src/algorithms/API.py`. Validate that replaying a warp matches the original aligned image within a documented float tolerance.
- Add `AlgorithmConfig` controls for method, memory budget and a simple smoothing strength. Keep low-level tuning private until measurements justify exposing it.
- Add method selection to CLI and GUI, progress for both passes, cancellation cleanup, and optional depth/confidence map export or internal inspection. Preserve the existing default and existing methods.

#### 2. Verify and tune against failure modes

- Synthetic tests with known source index maps: steps, slanted boundaries, one-pixel lines, crossing structures, low-texture areas, noise, saturated highlights, invalid warp borders, odd dimensions, 16-bit gradients, reordered inputs and >255 frames.
- End-to-end tests for aligned and stack-only modes, cancellation and rerun, CPU fallback, output shape/type and reproducibility. Verify that the index map refers to the actual source used at every unblended pixel.
- Compare score-only, temporal smoothing, spatial refinement and seam blending separately. Review 100% crops of halos and hairs; disable any step that helps averages but destroys fine foreground structures.
- Proposed release gate: on the curated real-stack set, a majority of blinded scene comparisons should prefer the candidate to current Depth Map; it must not show a repeatable regression against Pyramid on thin structures, and its CPU memory use must stay within the configured budget. Publish per-scene results, including failures. These are product acceptance criteria, not literature performance claims.

#### 3. Neural feasibility spike, then decide

- Reproduce V3 and FocusDeep inference in isolated environments with official code and weights, without copying them into the GPL app. Confirm code, weights and dataset rights before redistribution; public GitHub availability alone is insufficient.
- Test the same aligned scenes at native bit depth and resolution, including 100–150 frame input. Measure required conversion, tile seams, burst-length limits, GPU memory, CPU viability, model load time and deterministic replay. Compare against the new deterministic method and existing Pyramid, not solely against a paper's reported score.
- If a model clearly wins and is distributable, implement it as an optional backend with explicit model download/installation and a deterministic fallback. If it only wins on pairs or small synthetic stacks, keep it as a research result.

## Key risks and decision points

- **No captured sharp source:** Neither source selection nor ordinary fusion can recover genuinely missing detail. Flag low confidence or offer manual retouch; do not quietly invent pixels.
- **Alignment dominates fusion:** Focus breathing, parallax and motion can beat any focus estimator. If errors cluster around misregistered frames, test neighboring-frame registration and transform composition before changing fusion weights. FocusDeep also aligns successive frames for this reason. [Paper](https://arxiv.org/html/2311.17846).
- **Ambiguous focus maps:** Blur spread at a foreground edge can make a background frame score highly. Confidence gating, valid masks and hair-specific review are required; no global smoothing parameter will solve every case.
- **Memory and runtime:** Two-pass source replay trades I/O for bounded memory. Measure it on SSD and slower storage; use cached aligned tiles only if profiling shows a clear benefit.
- **Neural licensing and reproducibility:** Verify repository, model-weight and dataset terms separately before packaging. Keep optional dependencies out of the base install until the benchmark supports them.

## Deliverables

1. A landscape pair benchmark and independent alignment/PDRFuse/LightMFF reproduction reports, including overlay and mask diagnostics, 16-bit output checks, measured resource use and known failures.
2. A validated landscape alignment path and a two-frame `landscape` method, plus a tested N-way extension for three or four frames, with editable masks, CLI/GUI integration and user documentation.
3. A long-burst benchmark and separate `regularized_depth` CPU method with source-index and confidence diagnostics.
4. A neural long-burst feasibility report with a ship / do-not-ship recommendation based on real stacks, licenses and runtime.
