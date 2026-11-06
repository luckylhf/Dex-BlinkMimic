"""Render recorded CPU rollouts: reference, EMA and Live from one checkpoint."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dex_mimic.evaluation import render as original

W, H, FPS = 1920, 1080, 25
BG, TEXT = (10, 22, 35), (232, 237, 245)
COLORS = [(174, 186, 201), (65, 211, 172), (255, 161, 92)]
FONT = os.environ.get('DEX_VIDEO_FONT', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
NAMES = {'gbtc': 'GBTC 广播体操', 'dance_boy': 'dance-boy 舞蹈', 'dance_girl': 'dance-girl 长舞蹈'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def make(root, name, previews_only):
    out = root / 'videos'
    out.mkdir(exist_ok=True)
    original.CONFIG = {'transport': 'native'}
    contracts = [json.loads((root / actor / 'development/bundle/contract.json').read_text()) for actor in ('ema', 'live')]
    iteration = contracts[0]['iteration']
    assert contracts[1]['iteration'] == iteration
    runs = [original.read_run(root / actor, name, iteration) for actor in ('ema', 'live')]
    assert runs[0]['contract']['compiled_model_sha256'] == runs[1]['contract']['compiled_model_sha256']
    reference = root / 'references' / (name + '.npz')
    with np.load(reference, allow_pickle=False) as z:
        ref = {k: z[k].copy() for k in z.files if k != 'metadata'}
        meta = json.loads(str(z['metadata']))
    for run in runs:
        assert run['report']['reference']['source_sha256'] == meta['source_sha256']
        assert run['contract']['model_sha256'] == meta['model_sha256']
        assert run['contract']['joint_names'] == meta['joint_names']
        assert run['report']['evaluation_profile'] == 'strict_v1'
        assert run['report']['physical_support'] is False
    model_path = runs[0]['bundle'] / 'model.mjb'
    assert sha(model_path) == contracts[0]['compiled_model_sha256']
    model = mujoco.MjModel.from_binary_path(str(model_path))
    model.vis.global_.offwidth, model.vis.global_.offheight = 640, 480
    renderer = mujoco.Renderer(model, height=480, width=640)
    datas = [mujoco.MjData(model) for _ in range(3)]
    qadr = [model.jnt_qposadr[model.joint('robot/' + n).id] for n in meta['joint_names']]
    free = int(np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0])
    rq = int(model.jnt_qposadr[free])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 3.25, 130, -12
    fonts = {s: ImageFont.truetype(FONT, s) for s in (20, 23, 26, 28, 34)}
    last = len(ref['q']) - 1
    duration = last * .02
    paths = [ref['root_p'][:, :2]] + [r['poses'][:, rq:rq+2] for r in runs]
    all_xy = np.concatenate(paths)
    lo, hi = all_xy.min(0), all_xy.max(0)
    mid = (lo + hi) / 2
    plot = (65, 780, 835, 1030)
    scale = min((plot[2]-plot[0])/max(1., float(hi[0]-lo[0])+1.),
                (plot[3]-plot[1])/max(1., float(hi[1]-lo[1])+1.))
    for run in runs:
        metrics = run['report']['metrics']
        run['root_max'] = max(m['root_error_m'] for m in metrics)
        run['body_mean'] = float(np.mean([m['body_global_rmse_m'] for m in metrics]))
        run['joint_mean'] = float(np.mean([m['joint_rmse_rad'] for m in metrics]))

    def xy(v):
        return (int((plot[0]+plot[2])/2 + (v[0]-mid[0])*scale),
                int((plot[1]+plot[3])/2 - (v[1]-mid[1])*scale))

    def text(draw, pos, value, size=26, color=TEXT):
        draw.text(pos, value, font=fonts[size], fill=color)

    def frame(f):
        canvas = Image.new('RGB', (W, H), BG)
        draw = ImageDraw.Draw(canvas)
        text(draw, (24, 12), f'{NAMES[name]} | Dex-BlinkMimic · 保存点 #{iteration:,}', 34)
        text(draw, (1480, 20), f'同步时间 {f*.02:06.2f} 秒', 28)
        titles = ['参考动作（运动学）', f'EMA 策略 · 第 {iteration:,} 次', f'Live 策略 · 第 {iteration:,} 次']
        for col in range(3):
            x, d = col * 640, datas[col]
            idx, stopped = 0, False
            if col == 0:
                d.qpos[:] = runs[0]['poses'][0]
                d.qpos[qadr] = ref['q'][f]
                d.qpos[rq:rq+3] = ref['root_p'][f]
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, ref['root_R'][f].astype(np.float64).reshape(9))
                d.qpos[rq+3:rq+7] = quat
            else:
                run = runs[col-1]
                idx = min(f, len(run['poses'])-1)
                d.qpos[:] = run['poses'][idx]
                stopped = run['report']['failed'] and f >= len(run['poses'])-1
            mujoco.mj_forward(model, d)
            camera.lookat[:] = [d.qpos[rq], d.qpos[rq+1], .88]
            renderer.update_scene(d, camera)
            rendered = Image.fromarray(renderer.render().copy())
            if stopped:
                rendered = Image.blend(rendered, Image.new('RGB', rendered.size, (36, 13, 13)), .30)
            canvas.paste(rendered, (x, 112))
            draw.rectangle((x+1, 64, x+639, 111), fill=tuple(int(v*.22) for v in COLORS[col]))
            text(draw, (x+18, 70), titles[col], 23, COLORS[col])
            if col == 0:
                text(draw, (x+18, 604), f'完整参考：{duration:.2f} 秒', 26)
                text(draw, (x+18, 643), '同视角跟随各自骨盆；下图保留世界坐标', 23)
            else:
                run = runs[col-1]
                error = run['report']['metrics'][idx-1] if idx else {'root_error_m': 0., 'joint_rmse_rad': 0.}
                text(draw, (x+18, 604), f'当前位置偏差 {error["root_error_m"]:.2f} 米', 28, COLORS[col])
                text(draw, (x+18, 643), f'关节误差 {error["joint_rmse_rad"]:.3f} rad', 23)
                if stopped:
                    draw.rectangle((x+18, 492, x+622, 574), fill=(109, 28, 30))
                    text(draw, (x+32, 499), f'{run["report"]["elapsed_seconds"]:.2f} 秒触发终止', 28)
                    text(draw, (x+32, 538), '此处起定格，后续未继续仿真', 23)
        draw.line((0, 694, W, 694), fill=(61, 75, 94), width=2)
        text(draw, (24, 705), '骨盆轨迹俯视图 · 同一世界坐标 · 米', 26)
        text(draw, (24, 742), '灰：参考    绿：EMA    橙：Live', 20)
        draw.rectangle(plot, fill=(23, 32, 44), outline=(75, 87, 102))
        unit = 1 if max(hi-lo) < 5 else 5 if max(hi-lo) < 30 else 10
        xmin, xmax = mid[0]-(plot[2]-plot[0])/2/scale, mid[0]+(plot[2]-plot[0])/2/scale
        ymin, ymax = mid[1]-(plot[3]-plot[1])/2/scale, mid[1]+(plot[3]-plot[1])/2/scale
        for value in range(math.ceil(xmin/unit), math.floor(xmax/unit)+1):
            px = xy([value*unit, mid[1]])[0]
            draw.line((px, plot[1], px, plot[3]), fill=(43, 55, 70))
            text(draw, (px+3, plot[3]+2), str(value*unit), 20)
        for value in range(math.ceil(ymin/unit), math.floor(ymax/unit)+1):
            py = xy([mid[0], value*unit])[1]
            draw.line((plot[0], py, plot[2], py), fill=(43, 55, 70))
            text(draw, (plot[0]-40, py-13), str(value*unit), 20)
        for path, color in zip(paths, COLORS):
            points = [xy(v) for v in path[:min(f+1, len(path))]]
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            px, py = points[-1]
            draw.ellipse((px-5, py-5, px+5, py+5), fill=color)
        text(draw, (910, 712), '同一保存点 EMA 与 Live 对比；未筛选历史最佳', 23)
        for i, run in enumerate(runs):
            label = ['EMA', 'Live'][i]
            y = 758 + i*105
            outcome = '完整播放' if run['report']['completed'] else '触发终止'
            text(draw, (910, y), f'{label}：{run["report"]["elapsed_seconds"]:.2f} / {duration:.2f} 秒 · {outcome}', 28, COLORS[i+1])
            text(draw, (910, y+39), f'最大位置偏差 {run["root_max"]:.2f} 米 | 平均全身误差 {run["body_mean"]:.3f} 米', 23)
            text(draw, (910, y+70), f'平均关节误差 {run["joint_mean"]:.3f} rad（仅统计有效闭环区间）', 20)
        text(draw, (910, 976), 'CPU MuJoCo + ONNX；1 倍速；无支撑、无中途重置', 23)
        text(draw, (910, 1009), '严格终止：高度 / 倾斜超差即停止；终止后画面定格', 23, (255, 195, 135))
        text(draw, (910, 1042), '参考 30→50 Hz 重采样；本视频不代表真机验收', 20)
        return canvas

    stem = f'{name}_checkpoint_{iteration:06d}_reference_ema_live'
    previews = {0, min(250, last), min(460, last), last}
    previews |= {min(last, (len(r['poses'])+25)//2*2) for r in runs}
    try:
        if previews_only:
            for f in sorted(previews):
                frame(f).save(out / f'{stem}_{f:05d}.png')
            print('PREVIEWS_READY', name, flush=True)
            return
        path = out / (stem + '.mp4')
        with imageio.get_writer(path, fps=FPS, codec='libx264', quality=8, macro_block_size=2,
                                ffmpeg_params=['-movflags', '+faststart', '-pix_fmt', 'yuv420p', '-preset', 'veryfast', '-threads', '2']) as writer:
            for f in range(0, last+1, 2):
                im = frame(f)
                writer.append_data(np.asarray(im))
                if f in previews:
                    im.save(out / f'{stem}_{f:05d}.png')
                if f % 250 == 0:
                    print(name, 'rendered', f, '/', last, flush=True)
            im = frame(last)
            draw = ImageDraw.Draw(im)
            draw.rectangle((1430, 0, 1919, 62), fill=BG)
            text(draw, (1440, 15), f'回放结束 · {duration:.2f} 秒', 28)
            for _ in range(FPS):
                writer.append_data(np.asarray(im))
        evidence = dict(motion=name, checkpoint_iteration=iteration, comparison=['reference', 'ema', 'live'],
                        historical_best_evaluated=False, profile='strict_v1', transport='native CPU MuJoCo + ONNX CPU',
                        reference=str(reference), reference_sha256=sha(reference),
                        video=str(path), video_sha256=sha(path), fps=FPS, reference_seconds=duration,
                        end_hold_seconds=1, full_reference_rendered=True,
                        termination_display='freeze and label; no simulation after termination',
                        runs=[dict(actor=actor, report_path=r['report_path'], trajectory_sha256=r['trajectory_sha256'],
                                   completed=r['report']['completed'], elapsed_seconds=r['report']['elapsed_seconds'],
                                   termination_reason=r['report']['termination_reason'],
                                   max_root_error_m=r['root_max'], mean_body_global_rmse_m=r['body_mean'],
                                   mean_joint_rmse_rad=r['joint_mean']) for actor, r in zip(('ema', 'live'), runs)])
        (out / (stem + '.json')).write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + '\n')
        print('VIDEO_READY', path, flush=True)
    finally:
        renderer.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--preview-only', action='store_true')
    a = p.parse_args()
    for name in ('gbtc', 'dance_boy', 'dance_girl'):
        make(a.root.resolve(), name, a.preview_only)
