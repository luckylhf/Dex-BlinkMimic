"""Isolated GBTC initial-condition probes; never edits a reference or model."""
import argparse
import json
from pathlib import Path
import numpy as np
from dex_mimic.deployment.runtime import Controller, sha
from dex_mimic.deployment.native_mujoco import Plant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--old-reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    raw = args.study / 'references/gbtc.npz'
    rows = []
    probes = [
        ('latest_raw', 60, raw, 'reference_velocities'),
        ('latest_zero_initial_omega', 60, raw, 'zero_root_angular_velocity'),
        ('latest_zero_initial_velocities', 60, raw, 'zero_all_velocities'),
        ('latest_old_reference', 60, args.old_reference, 'reference_velocities'),
        ('half_hour_old_reference', 30, args.old_reference, 'reference_velocities'),
    ]
    for name, minute, reference, mode in probes:
        bundle = args.study / f'minute_{minute:03d}/evaluation/development/bundle'
        c = Controller(bundle, reference)
        p = Plant(c)
        if mode == 'zero_root_angular_velocity':
            p.data.qvel[p.rv + 3:p.rv + 6] = 0
        elif mode == 'zero_all_velocities':
            p.data.qvel[:] = 0
        import mujoco
        mujoco.mj_forward(p.model, p.data)
        initial = dict(root_velocity=p.data.qvel[p.rv:p.rv+3].tolist(),
                       root_angular_velocity_body=p.data.qvel[p.rv+3:p.rv+6].tolist(),
                       max_joint_velocity=float(abs(p.data.qvel[p.vadr]).max()))
        first_target = None
        for _ in range(c.frames - 1):
            target = c.step(*p.sensors())
            if first_target is None:
                first_target = dict(max_joint_target_change_rad=float(abs(target-p.data.qpos[p.qadr]).max()),
                                    max_action=float(abs(c.previous_action).max()))
            p.step(target)
            if p.measure(c.frame)['failed']:
                break
        report_path = args.out / f'{name}.json'
        p.save(report_path, 'isolated diagnostic; native CPU MuJoCo, same frozen EMA; one start, no reset, no support',
               dict(diagnostic_initial_velocity_mode=mode, reference_file_sha256=sha(reference),
                    initial_state=initial, first_policy_output=first_target))
        report = json.loads(report_path.read_text())
        rows.append(dict(name=name, iteration=12659 if minute == 60 else 11663,
                         reference=str(reference), reference_sha256=sha(reference),
                         policy_sha256=c.contract['policy_sha256'],
                         elapsed_seconds=report['elapsed_seconds'], completed=report['completed'],
                         mean_body_global_rmse_m=float(np.mean([e['body_global_rmse_m'] for e in p.errors])),
                         max_root_error_m=max(e['root_error_m'] for e in p.errors),
                         final_tilt_degrees=p.errors[-1]['tilt_degrees'],
                         initial_state=initial, first_policy_output=first_target,
                         report=str(report_path)))
    result = dict(probes=rows, interpretation='Old reference differs in motion preparation, not just initialization. Velocity-zero probes change only the simulator initial velocity; their reference goals remain unchanged. Diagnostic only, not replacement benchmark.')
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print('AUDIT_SUMMARY', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
