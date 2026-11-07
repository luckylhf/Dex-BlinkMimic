# Dex-BlinkMimic

基于 BlinkMimic 的 Dex 全身动作跟踪：导入机器人动作、训练策略、导出 ONNX，并通过 MuJoCo / ROS 回放和评测。训练核心来自 BlinkMimic `d264658`，Dex 资产与部署适配来自 Dex-SuperMimic。

人体 BVH / SMPL-X 先在 [Dex-Retarget](https://github.com/luckylhf/Dex-Retarget) 转换并验收，本项目读取其机器人动作导出。

## 文档导航

| 要做什么 | 文档 |
|---|---|
| 导入新动作、构建训练特征 | [数据导入](data/README.md) |
| 使用固定数据 Blink 配方 | [配方、环境检查与显存适配](deployment/README_recipe_20261107.md) |
| 评测 GBTC、dance-boy、dance-girl | [三动作冻结评测集](data/evaluation/supermimic_12341_v1/README.md) |
| 查看 luckylhf 路径和历史训练 | [部署与运行记录](deployment/README.md) |

## 安装与训练

训练使用 Linux / NVIDIA GPU 环境。以下命令在项目根目录执行：

```bash
uv sync --locked --extra train
uv run dex-mimic check
```

运行路径默认取当前目录；从其他目录调用时，设置 `DEX_MIMIC_ROOT` 和 `DEX_ASSETS`。

| 配置 | 用途 | 主要设置 |
|---|---|---|
| `configs/dex_blink_recipe.json` | 2026-11-07 固定数据实验 | 4096 环境、Muon、镜像比例 0.5、CUDA 图；关闭额外根部反馈和奖励 |
| `configs/dex_blink_safe.json` | 历史训练复现 | 8192 环境、Adam；启用根部反馈和奖励，关闭镜像、接触奖励、CUDA 图和探测 |

固定数据配方需要先准备 `compact.pt`，并通过[配方文档](deployment/README_recipe_20261107.md)中的环境和短测检查。训练输出使用新目录：

```bash
uv run dex-mimic train \
  --config configs/dex_blink_recipe.json \
  --compact-reference /path/to/frozen/compact.pt \
  --log_dir runs/new_fresh3500
```

完整训练数据、checkpoint 和 `test/` 结果未随仓库提供，须自行准备或从已有实验环境取得；仓库包含三动作冻结评测 NPZ。

## 导出与评测

导出 EMA 策略（导出 Live 策略时加 `--actor live`）：

```bash
uv run dex-mimic bundle export \
  --checkpoint /path/to/last.pt --out runs/new_bundle
```

策略输入为 `obs_history` 和 `goal`。导出目录记录关节顺序、观测布局、控制参数及模型/策略哈希；回放参考另用 `dex-mimic bundle motion` 准备。三动作评测命令见[评测集说明](data/evaluation/supermimic_12341_v1/README.md#运行评测)。

ROS Humble 运行环境支持 Python 3.10。已有 ROS 环境按基础依赖安装，训练依赖单独管理，并保留该环境已验证的 MuJoCo 版本。

## 已记录结果

2026-11-07 固定数据实验完成从头训练 3500 次，数据为 BONES 26,815 条和探索性 LAFAN 77 条。GBTC 和两条 LAFAN 走跑动作播放到结尾，高动态舞蹈仍提前终止，全局位置漂移明显。

本地完整结果包见 [结果、8 段视频与训练曲线](test/results/blink_recipe_fresh3500_20261107/README.md)（`test/` 被 Git 忽略）。该轮使用 11 月 7 日更新参考及独立评测历史，保留了原冻结基准。播放完成与轨迹跟踪精度分别报告；现有结果为仿真验证，真机验收仍需另做。

## 来源与许可

见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)、[DEX_THIRD_PARTY.md](DEX_THIRD_PARTY.md) 和 [LICENSE](LICENSE)。
