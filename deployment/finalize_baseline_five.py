"""Verify and collect the five frozen reference/policy baseline videos."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from dex_mimic.common import atomic, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    cfg = json.loads((root/'render_config.json').read_text())
    assert cfg['iteration'] == 12659 and len(cfg['motions']) == 5
    evidence_dir = root/'evidence'
    evidence_dir.mkdir(exist_ok=True)
    videos, metrics = [], {}
    for name, row in cfg['motions'].items():
        f = root/'videos'/f'{name}_checkpoint_012659_reference_policy.mp4'
        evidence = json.loads(f.with_suffix('.json').read_text())
        assert sha(f) == evidence['video_sha256']
        assert evidence['iteration'] == 12659
        assert evidence['policy_sha256'] == 'ff6bb9776b6c88195cb4ad3825d6422392eea3624ec20eab18a745ace57e7153'
        assert evidence['training_member'] == name.startswith(('walk','run'))
        probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=width,height,nb_frames,r_frame_rate:format=duration','-of','json',str(f)],text=True))
        stream = probe['streams'][0]
        assert stream['width'] == 1280 and stream['height'] == 960 and stream['r_frame_rate'] == '25/1'
        assert int(stream['nb_frames']) == evidence['frames']
        subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(f),'-f','null','-'],check=True)
        report_path = Path(row['evaluation_folder'])/(name+'_native.json')
        report = json.loads(report_path.read_text())
        assert report['physical_support'] is False and sha(report_path) == evidence['report_sha256']
        shutil.copy2(report_path, evidence_dir/report_path.name)
        trajectory = Path(report['trajectory'])
        shutil.copy2(trajectory, evidence_dir/trajectory.name)
        assert sha(evidence_dir/trajectory.name) == evidence['trajectory_sha256']
        metrics[name] = dict(reference_seconds=evidence['reference_seconds'],
            elapsed_seconds=report['elapsed_seconds'],completed=report['completed'],
            termination_reason=report['termination_reason'],training_member=evidence['training_member'],
            mean_body_global_rmse_m=sum(x['body_global_rmse_m'] for x in report['metrics'])/len(report['metrics']),
            max_root_error_m=max(x['root_error_m'] for x in report['metrics']))
        videos.append(dict(motion=name,path=str(f.relative_to(root)),sha256=sha(f),
            frames=evidence['frames'],seconds=float(probe['format']['duration']),full_decode_passed=True))
        print('BASELINE_VIDEO_VERIFIED',name,flush=True)
    atomic(root/'ready.json',dict(iteration=12659,actor='ema',evaluation_profile='jump_tracking_v2',
        reference_preprocessing='supermimic_12341_three_tests_v1',metrics=metrics,videos=videos,
        policy_sha256=evidence['policy_sha256'],checkpoint_sha256=evidence['checkpoint_sha256'],
        physical_support=False,mid_rollout_resets=False))
    lines=['# 第 12659 次模型：五动作留底视频','',
        '五段均为参考运动学 / 同一冻结 EMA 模型的真实 MuJoCo 闭环跟踪。用于持续接入新 BONES 训练前留底。', '',
        'GBTC、dance-boy、dance-girl 使用统一旧 SuperMimic 12341 参考。另两条沿用此前按行走/跑步代表性选出的 LAFAN walk1_subject5、run1_subject5；不是本轮重新随机抽取。三条用户测试动作未参与训练，两条 LAFAN 动作属于训练内，不能混作训练外泛化结论。', '',
        '| 视频 | 参考 / 秒 | 闭环 / 秒 | 平均全身误差 / cm | 结果 |',
        '|---|---:|---:|---:|---|']
    for v in videos:
        m=metrics[v['motion']]
        lines.append(f'| [{v["motion"]}]({v["path"]}) | {m["reference_seconds"]:.2f} | {m["elapsed_seconds"]:.2f} | {m["mean_body_global_rmse_m"]*100:.2f} | '+('完整播放' if m['completed'] else '倾斜超差终止')+' |')
    lines += ['', '视频：1280×960、25 FPS、1 倍速，完整参考持续展示，结尾保持 1 秒。相机跟随各自骨盆，下方轨迹保持同一世界坐标。无支撑、无中途重置，终止后定格并标明，未生成后续物理轨迹。', '',
        'jump_tracking_v2：高度超差仅记录，倾斜超差仍终止；完整播放不等于跟踪精度合格。平均误差只统计各自实际闭环区间。', '',
        '后续打包可直接使用 videos/ 内五个 MP4 及本 README；ready.json、各视频同名 JSON、SHA256SUMS.txt 和 evidence/ 保留来源、指标、真实轨迹与校验。这里没有附带或声称公开提供模型权重。五段均通过全帧解码、帧数和哈希检查。', '']
    (root/'README.md').write_text('\n'.join(lines))
    (root/'SHA256SUMS.txt').write_text(''.join(f'{v["sha256"]}  {v["path"]}\n' for v in videos))
    print('BASELINE_FIVE_READY',flush=True)


if __name__ == '__main__':
    main()
