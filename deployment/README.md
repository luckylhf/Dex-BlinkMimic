# luckylhf 部署与运行记录

远端项目目录：`/home/longhongfeng/Desktop/Dex-BlinkMimic`，环境为 `.venv`。部署对象是本项目的 Dex 适配版；内附 G1 源码快照保留作来源参考。

| 内容 | 入口 |
|---|---|
| 2026-11-07 固定数据 Blink 配方 | [配置与运行条件](README_recipe_20261107.md) |
| 导入新机器人动作 | [数据导入](../data/README.md) |
| 标准三动作参考、评测命令与历史隔离 | [冻结评测集](../data/evaluation/supermimic_12341_v1/README.md) |
| 2026-11-06 动态 BONES 续训方案（已停止） | [历史方案](LIVE_BONES_UNTIL_NOON.md) |

## 运行目录与状态

`current_run.txt` 等指针保存上次登记路径，不能单凭文件名判断进程仍在运行。到远端项目目录查看登记任务及进程：

```bash
cd /home/longhongfeng/Desktop/Dex-BlinkMimic
bash deployment/status.sh
run_dir=$(cat deployment/current_run.txt)
tail -f "$run_dir/stdout.log"
```

对已确认正在运行的任务，在对应目录写入 `STOP`，训练会在迭代边界保存并退出：

```bash
touch "$run_dir/STOP"
```

`environment_setup.log`、`environment_freeze.txt`、`preflight.json` 保存环境记录；每轮的 `args.json` 保存生效参数。16 GiB vGPU 的显存适配见配方文档，耗时以实际运行记录为准。

## 历史 LAFAN 试跑

以下均为 2026-11-06 的实验记录，复现使用 `configs/dex_blink_safe.json`。

数据目录 `data/lafan1/Retarget_061_全量77_20261106_095755` 保留原批次，训练副本为 `data/lafan1/motions_all77_exploratory/`：77 段、827,691 帧、50 Hz。原质量为 **2 accepted、75 review**，全量探索试跑保留这些标签。

accepted 动作采用原选定版本，review 动作取最后一次完整输出；共 v8 39 段、frax 23 段、hybrid 15 段。适配只整理字段及 XYZW → WXYZ 四元数顺序，逐段核对数组一致性，没有重新重定向。来源、选用版本和哈希见 `data_receipt.json`、`all77_data_receipt.json` 及每轮 `data_manifest.json`；质量报告在输入目录的 `evidence/`。

| 阶段 | 记录 |
|---|---|
| 仅两条 accepted 动作 | `runs/lafan_accepted_4096_20261106_120542`，完成 3500 次 |
| 全量 77 段、4096 环境保守试跑 | `runs/lafan_all77_exploratory_4096_20261106_132307`，第 963 次保存并停止 |
| 全量 77 段、原包 8192 环境配置 | 使用 `train_package_config.sh`，从随机初始化开始；后续续训单独留档 |
| 从第 12659 次接入动态 BONES | 已按用户要求停止；保存点及评测保留，原方案见上方链接 |

### 复现原包配置

`train_package_config.sh` 使用 safe 配置与训练代码默认值：8192 环境、3500 次迭代、buffer 512、策略/世界模型 batch 3072、horizon 16、Adam、BF16；启用根部反馈、EMA 和结构化世界模型。旧 `train_4096.sh` 对应较早的保守试跑。

具备上述远端环境、数据和 `.bootstrap/bin/uv` 后，确认旧任务已退出，再使用新目录启动：

```bash
run_dir="runs/lafan_all77_package8192_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"
cp deployment/all77_data_receipt.json "$run_dir/data_manifest.json"
printf '%s\n' "$run_dir" > deployment/current_run.txt
nohup bash deployment/train_package_config.sh "$run_dir" > "$run_dir/stdout.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$run_dir/train.pid"
```

脚本在运行目录保存配置快照和依赖列表。checkpoint 表明训练已运行，跟踪效果以相应评测结果为准。

## 历史适配检查

以下是旧环境的接口或短测记录，与完整训练结果分开解读：

| 检查 | 记录 |
|---|---|
| 64 环境、3 次更新，MuJoCo 3.16 | 结构化世界模型目标/积分往返、策略反传通过 |
| 动态 goal 的 ONNX 对照 | 64 个样本最大动作误差 `4.77e-7` |
| Muon 与同步探测 | 独立 3 次更新短测通过 |
| 有界分片切换 | 两分片、3 次更新短测通过；切换时清空旧帧索引的 replay，图与探测组合仍受代码限制 |
| Native / ROS | 三步物理轨迹逐元素一致，EMA / Live 导出通过共享历史评测入口 |

这些记录不代表完整锁定依赖栈、全部动作跟踪、训练吞吐或真机验收。11 月 7 日配方的独立环境检查、训练和评测结果见配方文档。
