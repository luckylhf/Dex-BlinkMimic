"""Verify final media, input/checkpoint identities and write a reviewable receipt."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
videos = root / 'videos'
receipt = json.loads((root / 'input_receipt.json').read_text())
summaries = {a: json.loads((root / a / 'summary.json').read_text()) for a in ('ema', 'live')}
iteration = summaries['ema']['iteration']
assert summaries['live']['iteration'] == iteration == 1899
checkpoint = root / 'checkpoint' / 'ckpt_001899.pt'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

assert sha(checkpoint) == '65e98b8b06e21f671b41830e405ac1d5f5bf96bf0184371384839ad644bb7d52'
media = []
for row in receipt['motions']:
    name = row['name']
    assert sha(Path(row['source'])) == row['source_sha256']
    assert sha(Path(row['motion'])) == row['motion_sha256']
    assert sha(Path(row['reference'])) == row['reference_sha256']
    p = videos / f'{name}_checkpoint_{iteration:06d}_reference_ema_live.mp4'
    evidence = json.loads(p.with_suffix('.json').read_text())
    assert sha(p) == evidence['video_sha256']
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_frames,pix_fmt:format=duration', '-of', 'json', str(p)], text=True))
    stream = probe['streams'][0]
    expected = len(range(0, row['frames'], 2)) + 25
    assert stream['codec_name'] == 'h264' and stream['width'] == 1920 and stream['height'] == 1080
    assert stream['r_frame_rate'] == '25/1' and stream['pix_fmt'] == 'yuv420p'
    assert int(stream['nb_frames']) == expected
    assert abs(float(probe['format']['duration']) - expected / 25) < .05
    subprocess.run(['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(p), '-f', 'null', '-'], check=True)
    media.append(dict(name=name, file=p.name, sha256=sha(p), bytes=p.stat().st_size,
                      reference_seconds=row['seconds'], video_seconds=float(probe['format']['duration']),
                      frames=expected, width=1920, height=1080, fps=25,
                      full_decode_passed=True, results=evidence['runs']))
    print('MEDIA_VERIFIED', name, expected, flush=True)

report = dict(checkpoint_iteration=iteration, checkpoint_sha256=sha(checkpoint),
              evaluation_profile='strict_v1', simulation='native CPU MuJoCo + ONNX CPU; not ROS or real robot',
              termination=dict(root_height_error_m=.25, root_vertical_component_difference=.8),
              physical_support=False, mid_rollout_resets=False, historical_best_evaluated=False,
              source_motions=receipt['motions'], videos=media,
              export_checks={a:json.loads((root/a/'development/bundle/export_check.json').read_text()) for a in ('ema','live')})
(videos / 'evaluation_summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
with (videos / 'metrics.csv').open('w') as f:
    writer = csv.writer(f)
    writer.writerow(['motion', 'actor', 'iteration', 'reference_seconds', 'elapsed_seconds', 'completed',
                     'termination_reason', 'max_root_error_m', 'mean_body_global_rmse_m', 'mean_joint_rmse_rad'])
    for row in media:
        for result in row['results']:
            writer.writerow([row['name'], result['actor'], iteration, row['reference_seconds'],
                             result['elapsed_seconds'], result['completed'], result['termination_reason'],
                             result['max_root_error_m'], result['mean_body_global_rmse_m'], result['mean_joint_rmse_rad']])
lines = ['# 第 1899 次 checkpoint：三个测试动作比对', '',
         '视频三栏依次为参考运动学、EMA 策略、同保存点 Live 策略。没有历史最佳筛选。', '',
         '固定 checkpoint SHA-256：`' + sha(checkpoint) + '`。评测开始时固定该快照，后续训练继续运行。', '',
         '| 动作 | 完整参考 | EMA 闭环时长 | Live 闭环时长 | EMA 最大位置偏差 |',
         '|---|---:|---:|---:|---:|']
for row in media:
    ema, live = row['results']
    lines.append(f'| [{row["name"]}]({row["file"]}) | {row["reference_seconds"]:.2f} 秒 | {ema["elapsed_seconds"]:.2f} 秒 | {live["elapsed_seconds"]:.2f} 秒 | {ema["max_root_error_m"]:.3f} 米 |')
lines += ['', 'GBTC 两种策略均播完；两段舞蹈均因高度偏差超过 0.25 米而提前终止。播完只说明未触发本协议的高度/倾斜停止条件，不代表位置或姿态跟踪精度已验收。', '',
          '输入为用户提供的 30 Hz、29 关节机器人 PKL。仅进行线性位置/关节插值、四元数 SLERP 到 50 Hz 及 xyzw→wxyz 字段适配；没有重新重定向，也没有将测试动作加入训练数据。', '',
          '原生 CPU MuJoCo 与 ONNX CPU 推理，50 Hz 控制，模型/关节合同匹配。每个动作仅从首帧初始化一次，无外部支撑、无中途重置。严格协议在根部高度误差 >0.25 米或根部旋转矩阵竖直分量差 >0.8 时终止。', '',
          '视频为 1920×1080、25 fps、1 倍速，无音轨；完整参考持续展示。策略终止后定格并标记，误差仅统计有效闭环区间。结束增加 1 秒定格。上方相机分别跟随各自骨盆，下方轨迹保留共同世界坐标并采用等比例坐标轴。', '',
          '指标与证据见 [evaluation_summary.json](evaluation_summary.json) 和 [metrics.csv](metrics.csv)。视频哈希与逐文件媒体检查见清单；三个视频均完成全帧解码验证。', '',
          '这是该中途保存点在指定动作和此仿真协议下的结果，不代表 ROS、真机、完整训练结束后的效果或所有动作的泛化表现。', '']
(videos / 'README.md').write_text('\n'.join(lines))
(videos / 'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(videos.glob('*.mp4'))))
print('EVALUATION_DELIVERABLES_READY', videos, flush=True)
