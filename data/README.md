# 动作导入与训练特征

新人体动作先在 [Dex-Retarget](https://github.com/luckylhf/Dex-Retarget) 的独立环境中使用 `dex-frax` 转换。本项目导入已验收的机器人动作，并构建 Blink 训练特征。

## 1. 选择转换结果

从单动作 `selection.json` 的 `selected_output`，或批次 `accepted.json` 的 `motions[].output`，定位实际求解阶段目录；相对路径以转换输出根目录为基准。

该阶段必须已完成、质量为 `accepted` 且无 flags，并包含：

```text
training.npz  motion.npz  run.json  identity.json  motion.json  quality.json
```

`review` 和 `preview_output` 保留作检查材料，不进入这条正式导入流程。历史探索数据继续保留原质量标签。

## 2. 编写导入清单

新建 `data/import_sources.jsonl`，每行一个 JSON 对象：

```json
{"id":"walk_01","source":"/path/to/accepted_stage","actor":"subject1","sampling_group":"locomotion","dataset":"lafan1"}
```

| 字段 | 要求 |
|---|---|
| `id` | 唯一动作名，使用字母、数字、下划线或连字符 |
| `source` | 求解阶段目录；相对路径按清单所在目录解释 |
| `actor` | 非空来源人物标识 |
| `sampling_group` | `dance`、`locomotion`、`stationary_upper` 或 `dynamic` |
| `dataset` | 可选，来源数据集名称 |
| `anchor` | 可选布尔值，默认 `false`，用于分片中的锚点动作 |
| `evaluation_only` | 可选布尔值，默认 `false`；评测动作单独列清单，建训练特征时会拒绝此类记录 |

## 3. 导入和使用

在项目根目录执行，输出目录须尚不存在：

```bash
uv run dex-mimic data import \
  --manifest data/import_sources.jsonl --out data/imported/walk_v1
```

导入器核对验收回执、转换指纹、文件哈希、模型、29 关节顺序、50 Hz、四元数约定及帧数，并逐元素比较两份 NPZ 的根位置、关节和旋转。`training.npz` 原样复制，不重采样或裁剪。

| 输出 | 内容 |
|---|---|
| `motions/<id>.npz` | 训练动作，可将 `motions/` 传给训练的 `--motion_dir` |
| `evidence/<id>/` | 四份 JSON 回执 |
| `records/`、`catalog.json` | 来源、文件哈希、质量标签和清单元数据 |

导入失败时目录保持 `status=importing`，不能建库；修正来源后换新目录重试。成功导入的副本可独立使用，无需保留转换机原路径。

需要旧式有界特征分片时，安装训练依赖后执行：

```bash
uv run dex-mimic data build \
  --records data/imported/walk_v1/records --out data/releases/walk_v1 --device cpu
```

构建前会再次核对导入副本与回执。固定全库 `compact.pt` 的准备方式见[固定数据配方](../deployment/README_recipe_20261107.md)，与上述分片流程分开使用。

转换端负责运动学验收；策略的动力学跟踪与平衡通过后续评测判断。本项目已移除原 `data convert` 和 `data check-speed` 入口。
