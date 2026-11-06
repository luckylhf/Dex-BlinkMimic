"""Reject mutations of already loaded references before touching training state."""
import json
from pathlib import Path
import tempfile
import unittest

from dex_mimic.common import sha
from dex_mimic.paths import ROOT
from dex_mimic.training.live_rotation import LiveRotation


class LiveGrowthTests(unittest.TestCase):
    def setUp(self):
        self.seed = ROOT/'data/mixed/lafan77_bones080_bounded_20261106'
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/'index.json'
        seed = json.loads((self.seed/'features/index.json').read_text())
        rows = seed['records']
        for row in rows:
            for key in ('pack', 'manifest'):
                row[key] = str((self.seed/row[key]).resolve())
        self.index = dict(seed_dataset_sha256=sha(self.seed/'manifest.json'), revision=0,
                          frame_capacity=seed['frame_capacity'], records=rows)
        self.save()
        self.rot = LiveRotation(self.seed, live_index=self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def save(self):
        self.path.write_text(json.dumps(self.index))

    def test_existing_pack_identity_is_immutable(self):
        previous = self.rot.index['records'][0]['pack_sha256']
        self.index['records'][0]['pack_sha256'] = 'changed'
        self.save()
        with self.assertRaisesRegex(ValueError, 'existing frozen shard'):
            self.rot._read_live_index()
        self.assertEqual(self.rot.index['records'][0]['pack_sha256'], previous)

    def test_index_cannot_remove_shards(self):
        self.index['records'].pop()
        self.save()
        with self.assertRaisesRegex(ValueError, 'append-only'):
            self.rot._read_live_index()

    def test_resident_capacity_cannot_change(self):
        self.index['frame_capacity'] += 1
        self.save()
        with self.assertRaisesRegex(ValueError, 'tensor capacity'):
            self.rot._read_live_index()


if __name__ == '__main__':
    unittest.main()
