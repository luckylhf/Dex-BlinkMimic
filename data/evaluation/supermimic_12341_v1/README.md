# 三动作冻结评测集

版本：`supermimic_12341_three_tests_v1`。本目录保存旧 Dex-SuperMimic 12341 的三份已处理动作和三份参考 NPZ，逐文件哈希见 [manifest.json](manifest.json)，评测入口见 [suite.json](suite.json)。六个 NPZ 与旧包逐字节一致，已纳入当前仓库。

## 参考处理

| 动作 | 50 Hz 开头裁剪 | 起步 | 结尾 | 参考时长 |
|---|---:|---|---|---:|
| GBTC | 27 帧 / 0.54 秒 | 1 秒站立 + 1 秒过渡 | 2.5 秒回站立 + 1 秒保持 | 18.92 秒 |
| dance-boy | 5 帧 / 0.10 秒 | 1 秒站立 + 1 秒过渡 | 2.5 秒回站立 + 1 秒保持 | 80.20 秒 |
| dance-girl | 不裁剪 | 保留原始开头 `original_start` | 无过渡 | 241.56 秒 |

裁剪在机器人 PKL 重采样为 50 Hz NPZ 后进行。GBTC、dance-boy 的保留起点是首个满足 `max_j(abs(q[i+1,j]-q[i,j])) * 50 < 1.5 rad/s` 的位置。这是单步关节速度启发式；表中裁剪量只对应这两份输入，不代表通用裁剪时长，也未检查根部角速度或连续稳定窗口。

站姿由匹配模型的正向运动学（FK）生成，按足底碰撞几何落地；位置和关节使用五次曲线过渡，旋转使用四元数 SLERP，随后计算速度。回放从最终参考首帧的位置、姿态和速度初始化。

评测直接复用冻结 NPZ。准备脚本核对 `test/videos/` 原始 PKL 的哈希后复制产物，不重新运行转换器，也不改原始 PKL / BVH。输入更换时须另建预处理版本和评测历史。

## 运行评测

在已安装训练依赖的项目根目录执行。须自备 checkpoint；准备脚本还需要匹配的 `test/videos/` 原始 PKL，`test/` 未随仓库提供。

一次准备并评测，使用新输出目录：

```bash
bash deployment/evaluate_three_test_pkls.sh \
  /path/to/ckpt.pt evaluations/new_three_test_review
```

仅准备输入：

```bash
.venv/bin/python deployment/prepare_three_test_pkls.py evaluations/new_three_test_inputs
```

已有本目录冻结参考时，可直接评测；省略 `--suite` 默认使用本集，无需原始 PKL：

```bash
uv run dex-mimic evaluate \
  --checkpoint /path/to/ckpt.pt --out evaluations/new_three_test_eval \
  --history evaluations/new_three_test_history --actor ema
```

## 结果与历史比较

默认 `jump_tracking_v2`：高度超差仅记录，倾斜超差仍终止。报告分别记录播放完成状态和跟踪质量；播放到结尾不代表轨迹精度合格，倾斜终止也不等同于确认摔倒。该协议不修改训练奖励或训练端终止条件。

不同参考、终止规则或控制条件使用独立 history，避免沿用旧输入的最佳结果。旧无过渡参考的复现需显式加 `--allow-legacy-three-tests`；原始适配器保留在 `deployment/prepare_three_test_pkls_raw_v1.py`。混合续训中的标准三动作也使用本集。

11 月 7 日固定数据实验使用了更新参考，其独立结果包见项目主 README；该实验未覆盖本冻结集。
