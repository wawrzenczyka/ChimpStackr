# ChimpStackr

![GitHub all releases](https://img.shields.io/github/downloads/noah-peeters/ChimpStackr/total) ![GitHub release (latest by date)](https://img.shields.io/github/downloads/noah-peeters/ChimpStackr/latest/total) ![GitHub](https://img.shields.io/github/license/noah-peeters/ChimpStackr) ![GitHub commits since latest release (by date)](https://img.shields.io/github/commits-since/noah-peeters/ChimpStackr/latest)

<p align="center">
  <img src="packaging/icons/chimpstackr_icon.png" width="200"/>
</p>

Open-source focus stacking application for Windows, macOS, and Linux.

## Features

- **9 stacking algorithms:** Laplacian Pyramid, Weighted Average, Depth Map, Exposure Fusion (HDR), Landscape, Landscape Blend, Near/Far Cut, Landscape Regions, Near/Far Depth
- **Automatic alignment:** Translation, Euclidean (rotation), Similarity (rotation + scale), Affine, landscape SIFT/ECC, or optional RoMa learned matches
- **Multiple results:** Select several methods and keep a separate result from each run for comparison and export
- **16-bit pipeline:** Full bit-depth preservation from RAW to output
- **Auto-crop:** Removes black edges from alignment shifts
- **Auto-tuning:** Parameters auto-detected from image resolution
- **GUI + CLI:** Full graphical interface and headless command-line tool
- **Cross-platform:** Native builds for Windows, macOS, Linux
- **Pause/resume/cancel:** Control long-running stacks
- **Before/after comparison:** Slider viewer for comparing input vs output
- **Drag & drop:** Drop image files or folders directly into the app

## Download

Pre-built packages are available on the [Releases](https://github.com/noah-peeters/ChimpStackr/releases) page:

| Platform | Download | Notes |
|---|---|---|
| **Windows** | `ChimpStackr-Windows.zip` | Extract and run `chimpstackr.exe` |
| **macOS** | `ChimpStackr-macOS.dmg` | Open DMG, drag to Applications |
| **Linux (AppImage)** | `ChimpStackr-Linux-x86_64.AppImage` | `chmod +x` and run |

## CLI Usage

The CLI allows headless focus stacking without a GUI:

```bash
# Basic stack
chimpstackr-cli --input images/*.jpg --output result.tif

# Align + stack with auto parameters
chimpstackr-cli -i images/*.jpg -o result.tif --align --auto --auto-crop

# Full options
chimpstackr-cli -i images/*.jpg -o result.png \
  --align \
  --method laplacian \
  --rotation-scale \
  --kernel-size 6 \
  --pyramid-levels 8 \
  --auto-crop \
  --quality-report

# Compare two landscape methods on the same pair; writes result_landscape.tif
# and result_landscape_blend.tif
chimpstackr-cli -i foreground.tif background.tif -o result.tif \
  --align --alignment-mode landscape \
  --method landscape,landscape_blend --bit-depth 16
```

**Available methods:** `laplacian` (default), `weighted_average`, `depth_map`, `exposure_fusion`, `landscape`, `landscape_blend`, `near_far_cut`, `landscape_regions`, `landscape_depth`. Repeat `--method` or separate method names with commas to produce separate files. `near_far_cut` accepts exactly two frames; the other landscape methods accept 2–4 frames.

## Stacking Algorithms

| Method | Best for | How it works |
|---|---|---|
| **Pyramid** | Fine detail (hairs, bristles, edges) | Laplacian pyramid decomposition, max-contrast selection per frequency band, local tone-mapping |
| **Weighted** | Smooth subjects, good color | Per-pixel contrast weighting with proper accumulation |
| **Depth Map** | Opaque surfaces, best color fidelity | Multi-scale sharpness with edge-aware bilateral smoothing |
| **HDR** | Varying exposure/lighting | Mertens exposure fusion (not for focus stacking) |
| **Landscape** | Two to four focus-bracketed landscape frames | Multi-scale focus evidence, guided region selection, original float32 source pixels |
| **Landscape Blend** | Two to four landscape frames with gentle focus transitions | The same focus map with narrow, confidence-gated seam blending |
| **Near/Far Cut** | Two frames with distinct near and far subjects | Edge-aware binary graph cut chooses coherent focused regions while preserving original pixels |
| **Landscape Regions** | Two to four landscape frames, including several near or far captures | Multi-label graph-cut selection favors coherent focus regions and retains clear small details |
| **Near/Far Depth** | Two to four frames with near objects against a distant scene | Depth Anything V2 Small proposes the foreground layer; focus evidence identifies near and far sources, then original pixels are composited automatically |

The two graph-cut methods use pinned PyMaxflow 1.3.2 and adapt established [graph-cut focus-segmentation ideas](https://journal.bit.edu.cn/zr/cn/article/doi/10.15918/j.tbit1001-0645.2015.06.017); they are automatic CPU alternatives, not trained SotA models. Near/Far Depth is our short-stack adaptation of the [Depth Anything V2 Small](https://github.com/DepthAnything/Depth-Anything-V2) depth estimator. The Apache-2.0 model code is vendored at pinned revision `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`; first use downloads its approximately 99 MB checkpoint and verifies SHA-256 before loading. Run `pixi run -e learned gui` (CPU) or `pixi run -e learned-cuda gui` (NVIDIA CUDA 12.4 wheels) to use it. A foreground bokeh halo can remain where neither source records clean background detail. Landscape alignment estimates a guarded SIFT similarity transform and refines it with ECC; when matches are unreliable, it keeps the input framing. Optional RoMa alignment uses learned matches to fit a guarded global similarity transform, falling back to landscape alignment when the fit is weak. Its first use downloads about 1.6 GB of pinned weights; CPU inference is slow. Published learned focus-fusion networks remain [research candidates](docs/stacking-research-plan.md) pending real-scene comparison.

## Build from Source

The Pixi environment uses Python 3.11–3.12 on Windows. The pip workflow retains the versions supported by `requirements.txt`.

### Pixi

Install [Pixi](https://pixi.sh/latest/#installation), then run commands from the project directory. Pixi creates and manages the project environment automatically:

```bash
# Run the GUI
pixi run gui

# Show CLI options, or run the CLI with arguments
pixi run cli-help
pixi run cli -- --input images/*.jpg --output result.tif

# Run the test suite in the environment that includes pytest
pixi run --environment dev test
```

The default environment contains the application runtime dependencies. The `dev` environment adds pytest. The current `pixi.toml` targets Windows; use the pip workflow on macOS or Linux. On Windows, use PowerShell syntax for paths and wildcard arguments as needed.

### pip / virtualenv

```bash
git clone https://github.com/noah-peeters/ChimpStackr.git
cd ChimpStackr
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run GUI
python src/run.py

# Run CLI
python -m src.cli --help

# Run tests
pip install pytest
pytest tests/ -v
```

## Packaging

Builds use PyInstaller with platform-specific post-processing. You can only build for your current platform.

```bash
# Install build tools
pip install pyinstaller

# Build (creates dist/chimpstackr/ and dist/ChimpStackr.app on macOS)
pyinstaller chimpstackr.spec --noconfirm

# Or use the platform scripts:
./scripts/build_macos.sh        # macOS → .dmg
./scripts/build_linux.sh        # Linux → .AppImage
.\scripts\build_windows.ps1     # Windows → .zip or installer
```

CI/CD automatically builds all platforms on tagged releases via GitHub Actions.

## Gallery

The following stacks were taken at ~4x magnification on a slightly wobbly rig (~150 images each), stacked with ChimpStackr and post-processed in [darktable](https://www.darktable.org/).

![Bij_TranslationAlignment](https://user-images.githubusercontent.com/17707805/196990942-413ea35c-2abb-4bce-9807-3f3d6b3de3c5.jpg)
![Edited](https://user-images.githubusercontent.com/17707805/196991117-dc4f1c76-cc87-4ef1-92ee-9a7484c7ff07.jpg)
![Bewerkt](https://user-images.githubusercontent.com/17707805/196996295-9fb6c365-ef10-4ef5-b451-1a7269156e53.jpg)

## Sources

- Focus stacking algorithm based on: Wang, W., & Chang, F. (2011). A Multi-focus Image Fusion Method Based on Laplacian Pyramid. *Journal of Computers*, 6(12).
- DFT image alignment adapted from: [imreg_dft](https://github.com/matejak/imreg_dft)
- Mertens exposure fusion: Mertens, T., Kautz, J., & Van Reeth, F. (2007). Exposure Fusion.
- Sum Modified Laplacian focus measure: Nayar, S.K., & Nakagawa, Y. (1994).

## License

GPL-3.0 - see [LICENSE](LICENSE) for details.
