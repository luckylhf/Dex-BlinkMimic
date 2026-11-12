# Dex 适配来源与许可

| 部分 | 来源 | 处理方式 |
|---|---|---|
| 训练核心 | BlinkMimic，基准提交 `d264658`；相关 InstantTrack / supermimic_vibed 代码来源见 `THIRD_PARTY_NOTICES.md` | 保留现有许可与归属，Dex 适配另行维护 |
| Dex 机器人及部署适配 | 既有 Dex-SuperMimic 工作树与 Dex 机器人资产 | 机器人资产保留 `assets/dex_evt/LICENSE`，不自动适用本项目代码的 MIT 许可 |
| 人体动作转换及运动学验收 | [luckylhf/Dex-Retarget](https://github.com/luckylhf/Dex-Retarget) | 外部独立项目；本包不分发其转换求解器或人体校准资源，仅导入已通过的机器人动作与来源回执 |
| 仿真与数值依赖 | mjlab、MuJoCo、MuJoCo Warp、Warp、PyTorch、NumPy 等 | 按 `pyproject.toml` / `uv.lock` 安装，适用各依赖自身许可 |
| BONES-SEED | [Bones Studio](https://bones.studio/)，BONES-SEED Dataset License Agreement | 下载、使用和成果发布按数据协议执行，训练成果署名见下文 |
| LAFAN1 | Ubisoft La Forge，CC BY-NC-ND 4.0 | 原始数据和本地转换产物按数据许可使用，公开分享改编材料须另获相应授权 |
| 其他人体输入 | 各输入提供方 | 输入、转换动作、视频与权重保留其来源和适用许可 |
| 三动作冻结评测 | GBTC、dance-boy、dance-girl 视频转换材料 | 六份机器人 NPZ 与预处理说明、清单和哈希随源码提供 |

Dex-Retarget 的上游方法、校准资源和许可由其仓库说明；训练包不再包含这些转换组件。保留的冻结数据和历史评测继续记录原始处理链，不能改写为本次新转换结果。

## BONES-SEED 成果署名

Training data includes Motion Data by Bones Studio. Use of the underlying dataset is subject to the BONES Motion Capture Dataset License Agreement. [Bones Studio](https://bones.studio/) · [BONES-SEED license](https://bones.studio/info/seed-license)

使用者直接向发布方获取数据并接受对应许可，按机构或企业资格确定使用范围。成果发布须满足协议中的署名、数据可提取性和分发条件。

## LAFAN1 数据使用

LAFAN1 动作与模型由 Ubisoft La Forge 提供，适用 CC BY-NC-ND 4.0。运行转换、训练或下载上游 G1 参考数据时，应核对对应数据和产物的许可。具体规则以 Ubisoft 官方协议及另行取得的授权为准。[LAFAN1 官方许可](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/license.txt)

## 视频动作处理工具

视频动作处理建议使用 GVHMR 提取人体运动、SMPL-X 构建人体关节，并通过 [Dex-Retarget](https://github.com/luckylhf/Dex-Retarget) 重定向到 Dex 机器人。项目署名及许可链接见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md#video-motion-processing-tools)。
