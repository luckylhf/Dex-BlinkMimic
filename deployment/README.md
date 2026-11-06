# luckylhf 训练部署

远端根目录：`/home/longhongfeng/Desktop/Dex-BlinkMimic`。

本目录部署的是外层 Dex 适配版，未部署内部原版 G1 包。独立环境为 `.venv`，依赖依据 `uv.lock` 安装；`environment_setup.log`、`environment_freeze.txt`、`preflight.json` 记录安装与启动检查。

## 数据

`data/lafan1/Retarget_061_全量77_20261106_095755` 是用户指定转换目录的完整副本。原目录保留。`data_receipt.json` 记录逐文件 SHA-256、全部 77 段覆盖记录以及本轮实际训练选用文件。

原批次为 2 段 accepted、75 段 review。用户已明确选择全部 77 段进行探索试跑。当前输入为 `data/lafan1/motions_all77_exploratory/`，共 827,691 帧、50 Hz，模型哈希和 29 关节顺序经训练读取器核验一致。

已验收动作保留原选定版本，review 动作取最后一次完整转换输出。选用版本为 v8 39 段、frax 23 段、hybrid 15 段。适配只整理数组字段并将四元数 xyzw 改为 wxyz，逐段验证位置、关节及四元数数据不变；没有重新重定向。`all77_data_receipt.json` 及训练目录内的 `data_manifest.json` 记录每段选用版本、源和输出哈希、原质量状态与全部 flags。输入目录内 `evidence/` 保留对应质量报告。原批次验收文件保持不变。

此前只使用 `run1_subject5`、`walk1_subject5` 的两段试跑已经完成 3500 次，产物保留在 `runs/lafan_accepted_4096_20261106_120542`；该训练不代表全量 77 段训练。随后全量 77 段的 4096 实例保守配置试跑在第 963 次保存并停止，目录为 `runs/lafan_all77_exploratory_4096_20261106_132307`。

## 训练

用户要求按包内原配置重来，当前使用 `deployment/train_package_config.sh`：全部训练参数直接采用 `configs/dex_blink_safe.json` 和训练代码默认值，命令只指定数据目录、运行目录和停止文件路径。从新模型初始化，不恢复旧 checkpoint。

原配置为 8192 个仿真实例、3500 次迭代、buffer_steps=512、pi_batch=3072、wm_batch=3072、horizon=16；代码默认 world model=2048×3、策略和 world model 使用 Adam，amp=bf16。root feedback、EMA 和 structured world model 按配置启用。代码默认每 50 次打印日志，每 2000 次或 3 分钟保存 checkpoint。启动目录中 `package_config.json` 保存本次配置快照，`args.json` 保存生效参数，`gpu_samples.csv` 记录显存采样。

旧的 `deployment/train_4096.sh` 仅用于复现此前的保守配置；当前入口以 `deployment/current_run.txt` 为准。

该远端是 16 GB vGPU，不支持 CUDA memory pool；启动脚本使用默认 PyTorch 分配器。该运行配置不能直接套用原版 G1 的 RTX 4090 训练耗时结论。

查看状态：

```bash
cd /home/longhongfeng/Desktop/Dex-BlinkMimic
bash deployment/status.sh
run_dir=$(cat deployment/current_run.txt)
tail -f "$run_dir/stdout.log"
```

请求当前训练在下一次迭代后保存并停止：

```bash
touch "$(cat deployment/current_run.txt)/STOP"
```

新训练使用新目录（运行前先确认旧进程已退出）：

```bash
run_dir="runs/lafan_all77_package8192_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"
cp deployment/all77_data_receipt.json "$run_dir/data_manifest.json"
printf '%s\n' "$run_dir" > deployment/current_run.txt
nohup bash deployment/train_package_config.sh "$run_dir" > "$run_dir/stdout.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$run_dir/train.pid"
```

训练更新及 checkpoint 仅证明训练已运行；策略跟踪效果需要另行评估或回放。

2026-11-06 三小时续训的评测已切换为 `jump_tracking_v2`：高度超差只记录，倾斜超差仍终止，报告分开记录完成状态与跟踪质量。训练端原包终止条件未修改。当前评测路径见 `current_progress_study.txt`；规则、迁移与回放验证见 [JUMP_EVALUATION.md](JUMP_EVALUATION.md)。旧严格协议结果保留在原目录。

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

2026-11-06 已启动从 12659 继续的 LAFAN + 动态 BONES 续训，截止 2026-11-07 12:00（上海时间），每小时保存五动作评测结果。动态接入、绝对截止和显存隔离说明见 [LIVE_BONES_UNTIL_NOON.md](LIVE_BONES_UNTIL_NOON.md)。
