# Third-party notices

Portions of `src/blinkmimic/` and `tools/` come from [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)
(Chong Zhang) and [InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack) (Huy-Quang-Dao), used under their MIT
Licenses, reproduced below.

## InstantTrack

```
MIT License

Copyright (c) 2026 Huy-Quang-Dao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## supermimic_vibed

```
MIT License

Copyright (c) 2026 ETH Zurich, Chong Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Dependencies and data

The Python dependencies (mjlab, MuJoCo, MuJoCo Warp, NVIDIA Warp, PyTorch, matplotlib, Pillow) are installed from
their own distributions under their own licenses (Apache-2.0, BSD-3-Clause and others) and are not redistributed here.
The LAFAN1 motions (Ubisoft La Forge, CC BY-NC-ND 4.0) and the Unitree G1 description (BSD-3-Clause) are downloaded
by `tools/download_data.sh` and are not redistributed here either; files derived from LAFAN1 (videos, GIFs,
checkpoints) are not covered by this repository's MIT License.

Local processing and publication of LAFAN-derived material follow the original dataset license and any separately
obtained authorization. Sharing adapted material requires separate authorization under the license's NoDerivatives
condition. [LAFAN1 official license](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/license.txt)

The evaluation motions in `data/evaluation/supermimic_12341_v1/` were produced by extracting human motion from
the GBTC / dance-boy / dance-girl videos and converting it to Dex robot motion. They are used for development evaluation.

The six evaluation NPZ files are supplied with the source release together with their preprocessing documentation,
manifests and hashes. Evaluation materials retain their source-specific permissions.

## Video motion processing tools

For video motion processing, the recommended tools are [GVHMR](https://github.com/zju3dv/GVHMR) for human motion
recovery, [SMPL-X](https://smpl-x.is.tue.mpg.de/) for body reconstruction, and
[Dex-Retarget](https://github.com/luckylhf/Dex-Retarget) for robot retargeting.

- GVHMR, Zhejiang University 3D Vision Group: World-Grounded Human Motion Recovery via Gravity-View Coordinates (2024).
  Its [software license](https://github.com/zju3dv/GVHMR/blob/main/LICENSE) permits educational, research and non-profit use.
- SMPL-X, Max Planck Institute for Intelligent Systems and collaborators: Expressive Body Capture: 3D Hands, Face,
  and Body from a Single Image (CVPR 2019). Model and software use is governed by the
  [non-commercial license](https://smpl-x.is.tue.mpg.de/modellicense.html).

## BONES-SEED attribution

Training data includes Motion Data by Bones Studio. Use of the underlying dataset is subject to the BONES Motion
Capture Dataset License Agreement. [Bones Studio](https://bones.studio/) · [BONES-SEED license](https://bones.studio/info/seed-license)

Obtain BONES-SEED directly from the publisher and accept its dataset license. Eligibility, attribution and distribution
conditions apply to dataset use and resulting models, software and experiment reports. This repository describes
aggregate training results and supplies training code. Raw BONES-SEED motions, dataset metadata and local training
exports remain in the user's data workspace. [Official dataset](https://huggingface.co/datasets/bones-studio/seed)
