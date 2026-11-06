"""Explicit dataset rotation and durable evaluation jobs around the baseline trainer."""
import atexit
import os
from pathlib import Path
import time
import torch
from dex_mimic.common import atomic
from .persistent_rotation import PersistentRotation
from .telemetry import Sampler

def add_arguments(parser):
    parser.add_argument('--config')
    parser.add_argument('--data-release', required=True)
    parser.add_argument('--switch-seconds', type=float, default=150)
    parser.add_argument('--switch-warmup-steps', type=int, default=64)
    parser.add_argument('--evaluation-minutes', type=float, default=30)
    parser.add_argument('--evaluation-queue')
    parser.add_argument('--stop-file')
    parser.add_argument('--skip-evaluation', action='store_true')

def initialize(args):
    log = Path(args.log_dir).resolve()
    queue = Path(args.evaluation_queue).resolve() if args.evaluation_queue else log/'evaluation_queue'
    queue.mkdir(parents=True, exist_ok=True)
    state, offset = None, 0.
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        state, offset = ck['reference_rotation'], ck['wall_time']
        del ck
    if os.environ.get('DEX_LIVE_REFERENCE_INDEX'):
        from .live_rotation import LiveRotation
        rot = LiveRotation(args.data_release, args.switch_seconds, start_index=state['shard'] if state else 0,
                           live_index=os.environ['DEX_LIVE_REFERENCE_INDEX'])
        if state and state.get('live_records'):
            rows = rot.index['records']
            if len(rows) < len(state['live_records']) or rows[:len(state['live_records'])] != state['live_records']:
                raise ValueError('Live resume index lost or changed previously adopted shards')
    else:
        rot = PersistentRotation(args.data_release, args.switch_seconds, start_index=state['shard'] if state else 0)
    if state:
        if state['dataset_sha256'] != rot.dataset_sha256:
            raise ValueError('Resume requires the same frozen dataset manifest')
        rot.cycles = state['cycles']
        rot.resume_generation = state['generation']
        rot.resume_delay = state.get('seconds_until_switch', args.switch_seconds)
        rot.bin_memory = state.get('difficulty_memory', {})
    if args.switch_warmup_steps < max(args.hist, args.wm_window):
        raise ValueError('Shard warmup must cover observation history and the world-model window')
    rot.switch_warmup_steps = args.switch_warmup_steps
    sampler = Sampler(log/'resources.jsonl').start()
    def cleanup():
        try:
            if hasattr(rot, 'coverage_bins'): rot.flush_coverage(log)
        finally:
            if hasattr(rot, 'prefetch'): rot.prefetch.close()
            sampler.close()
    atexit.register(cleanup)
    next_eval = 0.
    def checkpoint_ready(path, iteration, wall, final=False, baseline=False):
        nonlocal next_eval
        if os.environ.get('DEX_LIVE_REFERENCE_INDEX'):
            atomic(Path(path).with_suffix('.dataset.json'), dict(iteration=iteration,
                seed_dataset_sha256=rot.dataset_sha256, live_index_sha256=rot.live_digest,
                live_revision=rot.live_revision, shards=len(rot.index['records']),
                current_shard=rot.current, generation=rot.motion.generation,
                saved_epoch=time.time(), final=final))
        elapsed = max(0., wall-offset)
        if args.skip_evaluation or (elapsed < next_eval and not final and not baseline): return
        interval = args.evaluation_minutes * 60
        if interval <= 0: raise ValueError('evaluation-minutes must be positive')
        next_eval = (int(elapsed//interval)+1)*interval
        out = log/f'evaluation_{iteration:06d}'
        if not baseline:
            full = log/f'full_state_{iteration:06d}.pt'
            if not full.exists(): os.link(log/'last.pt', full)
        for actor in ('ema','live'):
            atomic(queue/f'job_{time.time_ns():020d}.json',dict(checkpoint=str(Path(path).resolve()),
                actor=actor,iteration=iteration,out=str(out/actor),label=log.name+' '+actor,
                elapsed_stage_seconds=elapsed,baseline=baseline,final=final))
    return rot, checkpoint_ready, lambda: bool(args.stop_file and Path(args.stop_file).exists())
