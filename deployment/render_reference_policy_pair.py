"""Render one reference and one recorded policy rollout, without simulation stepping."""
import argparse
import json
import math
import os
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dex_mimic.common import sha, atomic
from dex_mimic.evaluation import render as recorded

W, H, FPS = 1280, 960, 25
BG, TEXT, REF, POLICY = (15, 21, 31), (232, 237, 245), (169, 183, 204), (255, 161, 92)
FONT = os.environ.get('DEX_VIDEO_FONT', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')


def render(config, motion, preview_only=False):
    out = Path(config['out']); out.mkdir(parents=True, exist_ok=True)
    entry = config['motions'][motion]
    reference = Path(entry['reference'])
    recorded.CONFIG = {'transport': 'native'}
    run = recorded.read_run(Path(entry.get('evaluation_folder', config['evaluation_folder'])), motion, config['iteration'])
    report, contract, poses = run['report'], run['contract'], run['poses']
    assert report['evaluation_profile'] == config['evaluation_profile'] == 'jump_tracking_v2'
    assert contract['actor'] == 'ema' and contract['iteration'] == config['iteration']
    with np.load(reference, allow_pickle=False) as z:
        ref = {k: z[k].copy() for k in z.files if k != 'metadata'}
        metadata = json.loads(str(z['metadata']))
    source_sha = entry.get('reference_source_sha256', entry.get('training_sha256'))
    assert report['reference']['source_sha256'] == metadata['source_sha256'] == source_sha
    training_member = entry.get('training_member', True)
    assert metadata['model_sha256'] == contract['model_sha256']
    assert metadata['joint_names'] == contract['joint_names']
    assert sha(run['bundle']/'model.mjb') == contract['compiled_model_sha256']
    model = mujoco.MjModel.from_binary_path(str(run['bundle']/'model.mjb'))
    model.vis.global_.offwidth, model.vis.global_.offheight = 640, 480
    renderer = mujoco.Renderer(model, height=480, width=640)
    datas = [mujoco.MjData(model) for _ in range(2)]
    qadr = [model.jnt_qposadr[model.joint('robot/'+n).id] for n in metadata['joint_names']]
    free = int(np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0])
    rq = int(model.jnt_qposadr[free])
    camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 3.25, 130, -12
    fonts = {s:ImageFont.truetype(FONT, s) for s in (18, 20, 22, 24, 26, 30)}
    last = len(ref['q'])-1
    paths = [ref['root_p'][:, :2], poses[:, rq:rq+2]]
    all_xy = np.concatenate(paths); lo, hi = all_xy.min(0), all_xy.max(0); mid = (lo+hi)/2
    plot = (50, 733, 608, 923)
    scale = min((plot[2]-plot[0])/max(1., float(hi[0]-lo[0])+1.),
                (plot[3]-plot[1])/max(1., float(hi[1]-lo[1])+1.))
    body_mean = sum(x['body_global_rmse_m'] for x in report['metrics'])/len(report['metrics'])
    root_max = max(x['root_error_m'] for x in report['metrics'])

    def text(draw, pos, value, size=24, color=TEXT):
        draw.text(pos, value, font=fonts[size], fill=color)

    def xy(v):
        return (int((plot[0]+plot[2])/2+(v[0]-mid[0])*scale),
                int((plot[1]+plot[3])/2-(v[1]-mid[1])*scale))

    def frame(f):
        canvas = Image.new('RGB', (W,H), BG); draw = ImageDraw.Draw(canvas)
        text(draw, (18,9), entry['title'], 30)
        text(draw, (643,14), f'最终 EMA · 第 {config["iteration"]:,} 次', 24)
        text(draw, (1050,16), f'{f*.02:06.2f} 秒', 24)
        idx = min(f, len(poses)-1)
        stopped = bool(report['failed'] and f >= len(poses)-1)
        for col in range(2):
            x = col*640; d = datas[col]
            if col == 0:
                d.qpos[:] = poses[0]; d.qpos[qadr] = ref['q'][f]
                d.qpos[rq:rq+3] = ref['root_p'][f]
                quat = np.zeros(4); mujoco.mju_mat2Quat(quat, ref['root_R'][f].astype(np.float64).reshape(9))
                d.qpos[rq+3:rq+7] = quat
            else:
                d.qpos[:] = poses[idx]
            mujoco.mj_forward(model, d)
            camera.lookat[:] = [d.qpos[rq], d.qpos[rq+1], .88]
            renderer.update_scene(d, camera)
            im = Image.fromarray(renderer.render().copy())
            if col and stopped: im = Image.blend(im, Image.new('RGB', im.size, (36,13,13)), .3)
            canvas.paste(im, (x,96))
            color = POLICY if col else REF
            draw.rectangle((x+1,55,x+639,95), fill=tuple(int(v*.22) for v in color))
            text(draw, (x+16,59), '模型闭环跟踪' if col else '参考动作（运动学）', 24, color)
        err = report['metrics'][idx-1] if idx else dict(root_error_m=0.,joint_rmse_rad=0.,height_error_m=0.,height_violation=False)
        text(draw, (18,586), f'完整参考：{last*.02:.2f} 秒', 24)
        text(draw, (18,620), '相机各自跟随骨盆；下图保留共同世界坐标', 20)
        text(draw, (656,586), f'根部偏差 {err["root_error_m"]:.3f} 米', 24, POLICY)
        text(draw, (656,620), f'关节误差 {err["joint_rmse_rad"]:.3f} rad', 20)
        height_label = '高度超差' if err['height_violation'] else '高度偏差'
        text(draw, (980,620), f'{height_label} {err["height_error_m"]:.3f} 米', 20,
             POLICY if err['height_violation'] else TEXT)
        if stopped:
            draw.rectangle((658,476,1260,558), fill=(109,28,30))
            text(draw, (674,481), f'{report["elapsed_seconds"]:.2f} 秒 · 倾斜超差终止', 26)
            text(draw, (674,522), '此后定格；未继续仿真或中途重置', 20)
        draw.line((0,656,W,656), fill=(61,75,94), width=2)
        text(draw, (18,667), '骨盆轨迹 · 同一世界坐标 · 米', 24)
        text(draw, (18,704), '灰：参考     橙：最终 EMA', 18)
        draw.rectangle(plot, fill=(23,32,44), outline=(75,87,102))
        unit = 1 if max(hi-lo)<5 else 5
        x0,x1 = mid[0]-(plot[2]-plot[0])/2/scale, mid[0]+(plot[2]-plot[0])/2/scale
        y0,y1 = mid[1]-(plot[3]-plot[1])/2/scale, mid[1]+(plot[3]-plot[1])/2/scale
        for v in range(math.ceil(x0/unit),math.floor(x1/unit)+1):
            px=xy([v*unit,mid[1]])[0];draw.line((px,plot[1],px,plot[3]),fill=(43,55,70));text(draw,(px+2,925),str(v*unit),18)
        for v in range(math.ceil(y0/unit),math.floor(y1/unit)+1):
            py=xy([mid[0],v*unit])[1];draw.line((plot[0],py,plot[2],py),fill=(43,55,70));text(draw,(5,py-10),str(v*unit),18)
        for path,color in zip(paths,(REF,POLICY)):
            points=[xy(v) for v in path[:min(f+1,len(path))]]
            if len(points)>1:draw.line(points,fill=color,width=3)
            px,py=points[-1];draw.ellipse((px-4,py-4,px+4,py+4),fill=color)
        outcome = ('完整播放 · 有跟踪超差' if not report['tracking_criteria_passed'] else '完整播放') if report['completed'] else '倾斜超差终止'
        text(draw,(660,672),f'闭环 {report["elapsed_seconds"]:.2f} / {last*.02:.2f} 秒',26,POLICY)
        text(draw,(660,714),outcome,24,POLICY)
        text(draw,(660,756),f'平均全身误差 {body_mean:.3f} 米',22)
        text(draw,(660,792),f'最大根部偏差 {root_max:.3f} 米',22)
        text(draw,(660,835),'原生 MuJoCo；1 倍速；无支撑、无重置',20)
        text(draw,(660,870),entry.get('membership_note', '训练内动作；原批次转换质量 accepted'
                                     if training_member else '训练外固定测试；旧 SuperMimic 参考'),20)
        text(draw,(660,905),'高度超差只记录；倾斜超差仍终止',20,POLICY)
        return canvas

    stem = motion+f'_checkpoint_{config["iteration"]:06d}_reference_policy'
    previews=sorted(set((0,min(250,last),min(1000,last),min(len(poses)+1,last),last)))
    try:
        if preview_only:
            for f in previews:frame(f).save(out/f'{stem}_{f:05d}.png')
            return
        output=out/(stem+'.mp4')
        with imageio.get_writer(output,fps=FPS,codec='libx264',quality=8,macro_block_size=2,
            ffmpeg_params=['-movflags','+faststart','-pix_fmt','yuv420p','-preset','veryfast','-threads','2']) as writer:
            for f in range(0,last+1,2):
                im=frame(f);writer.append_data(np.asarray(im))
                if f in previews:im.save(out/f'{stem}_{f:05d}.png')
                if f%500==0:print(motion,'rendered',f,'/',last,flush=True)
            im=frame(last);draw=ImageDraw.Draw(im);draw.rectangle((1035,0,1279,53),fill=BG)
            text(draw,(1040,16),'回放结束',24)
            for _ in range(FPS):writer.append_data(np.asarray(im))
        evidence=dict(motion=motion,iteration=config['iteration'],actor='ema',evaluation_profile=config['evaluation_profile'],
            video=str(output),video_sha256=sha(output),reference=str(reference),reference_sha256=sha(reference),
            reference_source_sha256=source_sha,training_sha256=source_sha if training_member else None,
            training_member=training_member,original_quality_status=entry.get('original_quality_status', 'accepted' if training_member else None),
            report_sha256=sha(run['report_path']),trajectory_sha256=run['trajectory_sha256'],
            checkpoint_sha256=contract['checkpoint_sha256'],policy_sha256=contract['policy_sha256'],
            width=W,height=H,fps=FPS,reference_seconds=last*.02,frames=len(range(0,last+1,2))+FPS,
            completion=report['completed'],termination_reason=report['termination_reason'],
            camera='same camera orientation/distance, each follows own pelvis; fixed equal-scale world XY plot',
            termination_display='freeze and label; no simulation after recorded termination',end_hold_seconds=1)
        atomic(out/(stem+'.json'),evidence)
        print('VIDEO_READY',output,flush=True)
    finally:
        renderer.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--motion');parser.add_argument('--preview-only',action='store_true')
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    for motion in ([args.motion] if args.motion else config['motions']):render(config,motion,args.preview_only)
