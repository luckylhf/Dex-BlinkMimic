# Dex-BlinkMimic

Dex whole-body tracking adapted from BlinkMimic d264658. Dex robot assets and deployment originate from the Dex-SuperMimic working tree. **Human motion must be converted with [luckylhf/Dex-Retarget](https://github.com/luckylhf/Dex-Retarget).** This package imports its accepted robot-motion exports and builds training features; it does not bundle a retargeting solver.

## Verified so far

Isolated 64-environment, three-update GPU smoke on the existing MuJoCo 3.16 stack. Structured world-model target/integration round-trip, actor backward and dynamic-goal ONNX equivalence passed. Maximum ONNX action error was 4.77e-7 over 64 samples. These are engineering checks. Controller performance and Blink throughput remain untested.

uv.lock pins the separate Blink training stack. Exact-stack validation is pending. Existing checkpoints belong to their original architectures and optimizers.

## Entry points

Run from this directory or set DEX_MIMIC_ROOT and DEX_ASSETS.

    uv sync --locked --extra train
    uv run dex-mimic train --config configs/dex_blink_safe.json --motion_dir /qualified/motions --log_dir runs/smoke
    uv run dex-mimic bundle export --checkpoint runs/smoke/last.pt --out runs/smoke/bundle

The exported policy accepts obs_history and goal. Motions are prepared separately, using the bundle motion subcommand. Each bundle records joint order, observation layout, actuator settings, policy/model hashes and MuJoCo version.

## Integration status

- Blink structured world model, Muon and probe reset algorithms are retained.
- Dex root feedback and root reward can be enabled from the first update.
- NPZ inputs validate robot identity, joint order, quaternion convention and 50 Hz.
- sampling_weights.json supplies source weights through bin_prior.
- Explicit Dex mirror partners are configured. Small-pose FK tests found 0.95 mm geometric asymmetry. Mirror augmentation stays disabled pending inertia/contact symmetry checks.
- contact_z=0.06 is provisional. Contact shaping stays disabled pending geometric calibration.
- The safe recipe disables CUDA graphs and asynchronous probes pending exact-stack reset/replay verification. Muon plus synchronous probe sweeps completed a separate three-update smoke on the existing stack. Muon plus synchronous probe sweeps completed a separate three-update smoke on the existing stack.
- Bounded dataset loading, source difficulty memory, coverage and shard-loop integration passed a two-shard, three-update smoke on the existing stack. Changing source invalidates replay, including device counters. Shard mode explicitly gates CUDA graphs and probe sweeps pending joint lifetime verification.
- New human motions are converted and assessed in Dex-Retarget. This package verifies its completed acceptance receipts, model identity, hashes and motion-array parity before importing training NPZs. See [data/README.md](data/README.md).

## Sources and licenses

See THIRD_PARTY_NOTICES.md and DEX_THIRD_PARTY.md for algorithm and code attribution.

The runtime package supports Python 3.10 for ROS Humble. Training dependencies are optional and isolated. For an existing ROS environment, install the package with pip without the train extra, preserving its validated MuJoCo version.

Three-step native/ROS physical trajectories were elementwise identical on the existing deployment stack. Explicit EMA and live exports both passed the shared-history evaluation entry point. Runtime exports, reference preparation and ROS execution are validated at interface-smoke scope.

## Motion conversion

Use **[github.com/luckylhf/Dex-Retarget](https://github.com/luckylhf/Dex-Retarget)** for BVH or supported SMPL-X human inputs. Follow that repository's installation and conversion instructions in a separate environment. The command remains `dex-frax`; choose solver modes there. No converter installation or solver import is required in the Blink training environment.

After Dex-Retarget reports `accepted`, point an import manifest at the selected stage output directory containing `training.npz`, `motion.npz`, `run.json`, `identity.json`, `motion.json` and `quality.json`. Run `dex-mimic data import`, then optionally `dex-mimic data build`; complete examples are in [data/README.md](data/README.md). The former `data convert` and `data check-speed` entries have been removed. Conversion and kinematic acceptance belong to Dex-Retarget; policy tracking and dynamics require separate evaluation.

## 三个固定动作的统一评测口径（2026-11-06）

后续 GBTC、dance-boy、dance-girl 采用旧 Dex-SuperMimic 12341 的逐动作处理方式，版本为 `supermimic_12341_three_tests_v1`。三段的处理并不相同：

| 动作 | 50 Hz NPZ 开头裁剪 | 起步 | 结尾 | 完整参考时长 |
|---|---:|---|---|---:|
| GBTC | 27 帧 / 0.54 秒 | 1 秒站立 + 1 秒平滑过渡 | 2.5 秒回站立 + 1 秒保持 | 18.92 秒 |
| dance-boy | 5 帧 / 0.10 秒 | 1 秒站立 + 1 秒平滑过渡 | 2.5 秒回站立 + 1 秒保持 | 80.20 秒 |
| dance-girl | 不裁剪 | `original_start` 原始开头，不加站立过渡 | 不添加过渡 | 241.56 秒 |

裁剪发生在机器人 PKL 重采样为 50 Hz NPZ 之后，原始 PKL/BVH 不修改，也不重新重定向。旧工具对 GBTC、dance-boy 取首个 `max_j(abs(q[i+1,j]-q[i,j])) * 50 < 1.5 rad/s` 的位置作为保留起点；0.54 秒和 0.10 秒是对应文件的实际结果，并非所有动作的固定裁剪时长。这是单步关节速度启发式，不检查根部角速度，不要求连续稳定窗口，不能将裁掉的所有帧都视为错误动作。

站姿由匹配模型 FK 生成并按足底碰撞几何落地；关节/根位置使用五次曲线过渡，旋转使用四元数 SLERP。旧输出在加入过渡后计算速度；回放按最终参考首帧的位置、姿态和速度初始化，不额外偷偷清零速度。

本地与 luckylhf 均提供 `data/evaluation/supermimic_12341_v1/`，含三条旧版已处理动作、参考、`manifest.json` 和 `suite.json`，六个 NPZ 与旧包逐字节相同。固定评测直接复用这些冻结产物，避免重新插值产生漂移；**不是每次重新执行转换器**。清单同时锁定本项目 `test/videos/` 三个原始 PKL 的 SHA-256。输入变化时准备脚本拒绝套用旧参考，应另建有来源证据的预处理版本。它是这三条已知动作的固定基准，不是任意新 PKL 的通用裁剪转换器。

统一入口（项目目录，现有 `.venv`）：

```bash
export DEX_MIMIC_ROOT="$PWD" DEX_ASSETS="$PWD/assets/dex_evt" PYTHONPATH="$PWD/src"
# 一次准备、评测，必须给新的输出目录；不会启动训练。
bash deployment/evaluate_three_test_pkls.sh /path/to/ckpt.pt evaluations/new_three_test_review

# 或仅准备冻结输入：
.venv/bin/python deployment/prepare_three_test_pkls.py evaluations/new_three_test_inputs
# 通用评测省略 --suite 时也默认使用这一冻结集：
.venv/bin/python -m dex_mimic.evaluation.evaluate \
  --checkpoint /path/to/ckpt.pt --out evaluations/new_three_test_eval \
  --history evaluations/new_three_test_history --actor ema
```

默认评测继续采用 `jump_tracking_v2`：高度超差仅记录，倾斜超差仍终止；不修改训练奖励、训练数据或训练端终止条件。当前三动作入口见 `deployment/current_three_test_suite.txt`。混合数据续训启动器也从该标准集复制三动作参考；有界续训复用基线前会核对参考一致性。

已完成的历史训练、原始 PKL 无过渡评测、最佳记录和视频保留。新参考必须使用新评测目录和新 history；历史最佳不能跨输入哈希直接沿用。通用评测器会拒绝以这三个标准动作名传入旧的无过渡参考；如仅复现历史或诊断，需明确加 `--allow-legacy-three-tests` 并使用独立 history。旧原始适配器保留为 `deployment/prepare_three_test_pkls_raw_v1.py`，仅供此类显式复现，不是默认入口。协议说明和哈希不代表真机验证。

2026-11-06 已启动从 12659 继续的 LAFAN + 动态 BONES 续训，截止 2026-11-07 12:00（上海时间），每小时保存五动作评测结果。动态接入、绝对截止和显存隔离说明见 [LIVE_BONES_UNTIL_NOON.md](deployment/LIVE_BONES_UNTIL_NOON.md)。
