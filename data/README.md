# Dex-Retarget 动作导入与训练特征

**所有新人体动作必须使用 [github.com/luckylhf/Dex-Retarget](https://github.com/luckylhf/Dex-Retarget) 进行转换。** 在该项目的独立环境安装并运行 `dex-frax`，按其 README 选择输入格式与求解模式。本项目不包含重定向求解器，不接收原始人体 BVH/SMPL-X 作为训练输入，也不在训练环境调用转换程序。

## 导入已通过的结果

在 Dex-Retarget 的 `selection.json` 中找到 `selected_output`，或在批次的 `accepted.json` 中找到对应动作的 `output`，相对于该批次输出根目录定位实际求解阶段目录。只导入已完成、`accepted` 且没有 flags 的结果；`preview_output` 和 `review` 结果不作为已通过的训练数据。

每个目录应包含 `training.npz`、`motion.npz`、`run.json`、`identity.json`、`motion.json` 和 `quality.json`。新建 JSONL 清单，例如 `data/import_sources.jsonl`（每行一条）：

```json
{"id":"walk_01","source":"/path/to/accepted_stage","actor":"subject1","sampling_group":"locomotion","dataset":"lafan1"}
```

- `source` 是上述实际求解阶段目录，不是原始人体文件、批次根目录或单个 NPZ。相对路径按清单所在目录解释。
- `id` 唯一，仅用字母、数字、下划线或连字符；`actor` 为非空来源人物标识。
- `sampling_group` 可用 `dance`、`locomotion`、`stationary_upper`、`dynamic`。
- `dataset` 可选，用于来源标注；`anchor` 和 `evaluation_only` 可选，默认 `false`。
- 训练与评测使用不同清单；`data build` 拒绝包含 `evaluation_only=true` 的记录。

从 Dex-BlinkMimic 根目录运行（或使用已安装的同名命令）：

```sh
export DEX_MIMIC_ROOT="$PWD" DEX_ASSETS="$PWD/assets/dex_evt" PYTHONPATH="$PWD/src"
python -m dex_mimic.cli data import --manifest data/import_sources.jsonl --out data/imported/walk_v1
# 可直接将 data/imported/walk_v1/motions 交给 --motion_dir。
# 需要有界特征分片时，安装训练依赖后执行：
python -m dex_mimic.cli data build --records data/imported/walk_v1/records --out data/releases/walk_v1 --device cpu
```

输出目录必须是新目录，拒绝覆盖。导入失败留下 `status=importing` 的目录不能建库；处理原因后使用新的输出目录。`data convert` 和 `data check-speed` 已移除；转换与运动学质量审核由 Dex-Retarget 承担。

## 核验与产物

导入核对 Dex-Retarget 的完成/验收状态、无 flags、转换指纹、生产端文件哈希、模型 SHA-256、29 关节顺序、50 Hz、四元数约定、帧数和有限数值，并逐元素核对 `training.npz` 与 `motion.npz` 的根位置、关节和四元数。

`training.npz` 按原文件复制到 `motions/<id>.npz`，不重新重定向、不重采样、不裁剪。`evidence/<id>/` 保存四份 JSON 回执；`records/` 和 `catalog.json` 锁定来源、文件哈希、原始质量结果及清单元数据。构建特征前再次核对导入副本与回执，不要求转换程序安装在训练机，也不要求保留转换机上的原路径。

既有训练数据、固定三动作评测、历史训练和转换收据保留各自原始来源与质量状态；历史探索数据不能因本次代码更新被改称 accepted。运动学通过不等于策略跟踪、动力学平衡或真机验证完成。
