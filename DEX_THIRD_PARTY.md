# Dex 适配来源与许可

| 部分 | 来源 | 处理方式 |
|---|---|---|
| 训练核心 | BlinkMimic，基准提交 `d264658`；相关 InstantTrack / supermimic_vibed 代码来源见 `THIRD_PARTY_NOTICES.md` | 保留现有许可与归属，Dex 适配另行维护 |
| Dex 机器人及部署适配 | 既有 Dex-SuperMimic 工作树与 Dex 机器人资产 | 机器人资产保留 `assets/dex_evt/LICENSE`，不自动适用本项目代码的 MIT 许可 |
| 人体动作转换及运动学验收 | [luckylhf/Dex-Retarget](https://github.com/luckylhf/Dex-Retarget) | 外部独立项目；本包不分发其转换求解器或人体校准资源，仅导入已通过的机器人动作与来源回执 |
| 仿真与数值依赖 | mjlab、MuJoCo、MuJoCo Warp、Warp、PyTorch、NumPy 等 | 按 `pyproject.toml` / `uv.lock` 安装，适用各依赖自身许可 |
| BONES-SEED / LAFAN1 及其他人体输入 | 各原始数据集或输入提供方 | 输入、转换动作、视频与权重保留其来源和适用许可，不因导入而重新授权 |

Dex-Retarget 的上游方法、校准资源和许可由其仓库说明；训练包不再包含这些转换组件。保留的冻结数据和历史评测继续记录原始处理链，不能改写为本次新转换结果。
