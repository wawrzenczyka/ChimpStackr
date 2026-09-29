# Depth Anything V2 Small source snapshot

This directory contains the Python inference modules needed for the Small model from
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), pinned to
commit `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`. The files are copied without
behavioral changes. The upstream Apache-2.0 license is included in [LICENSE](LICENSE).

The checkpoint is downloaded separately from the
[official Small model repository](https://huggingface.co/depth-anything/Depth-Anything-V2-Small).
`depth_guided.py` verifies its SHA-256 hash before loading it.
