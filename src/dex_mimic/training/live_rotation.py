"""Append-only, verified reference growth at the existing shard-switch boundary."""
import json
import hashlib
from pathlib import Path
import time

from dex_mimic.common import sha
from dex_mimic.training.persistent_rotation import PersistentRotation
from dex_mimic.training.prefetch import PackPrefetch


class LiveRotation(PersistentRotation):
    def __init__(self, *args, live_index, **kwargs):
        requested_start = kwargs.get('start_index', 0)
        super().__init__(*args, **kwargs)
        self.live_index = Path(live_index)
        self.live_digest = None
        self.live_revision = 0
        self._read_live_index()
        self.current = requested_start % len(self.index['records'])

    def _read_live_index(self):
        raw = self.live_index.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest == self.live_digest:
            return False
        value = json.loads(raw)
        if value['seed_dataset_sha256'] != self.dataset_sha256:
            raise ValueError('Live index belongs to a different seed release')
        if value['frame_capacity'] != self.index['frame_capacity']:
            raise ValueError('Live growth cannot change resident tensor capacity')
        rows = value['records']
        if len(rows) < len(self.index['records']):
            raise ValueError('Live reference index must be append-only')
        for row in rows:
            for key in ('pack', 'manifest'):
                if not Path(row[key]).is_absolute():
                    raise ValueError('Live index paths must be absolute')
        for old, new in zip(self.index['records'], rows):
            for key in ('pack', 'manifest', 'pack_sha256', 'manifest_sha256'):
                if old[key] != new[key]:
                    raise ValueError('An existing frozen shard was changed')
        self.index = dict(self.index, records=rows)
        self.live_digest = digest
        self.live_revision = value['revision']
        return True

    def before_iteration(self, ns, iteration):
        # Reload only at a boundary that already invalidates replay and refills it.
        if self.next_at is not None and time.monotonic() >= self.next_at:
            try:
                changed = self._read_live_index()
            except Exception as error:
                # A bad update must not interrupt learning on the last valid index.
                changed = False
                with (Path(ns['args'].log_dir) / 'live_index_errors.jsonl').open('a') as f:
                    f.write(json.dumps(dict(time=time.time(), error=repr(error)))+'\n')
            if changed:
                self.prefetch.close()
                self.prefetch = PackPrefetch(self.index, self.motion.validate_pack)
                self.prefetch.start((self.current + 1) % len(self.index['records']))
                with (Path(ns['args'].log_dir) / 'live_adoptions.jsonl').open('a') as f:
                    f.write(json.dumps(dict(time=time.time(), iteration=iteration, revision=self.live_revision,
                                            shards=len(self.index['records']), index_sha256=self.live_digest))+'\n')
        super().before_iteration(ns, iteration)

    def checkpoint_state(self):
        state = super().checkpoint_state()
        state.update(live_index=str(self.live_index), live_index_sha256=self.live_digest,
                     live_revision=self.live_revision, live_records=self.index['records'])
        return state
