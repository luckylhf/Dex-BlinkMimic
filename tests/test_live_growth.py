"""Live-index metadata validation with temporary fixtures, without GPU imports.

Load the real LiveRotation class with inert parent/prefetch dependencies: these
checks call only _read_live_index, never tensor loading, rotation or simulation.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from dex_mimic.common import sha


def live_rotation_class():
    dependencies = {}
    for module, attribute in (
        ('dex_mimic.training.persistent_rotation', 'PersistentRotation'),
        ('dex_mimic.training.prefetch', 'PackPrefetch'),
    ):
        stub = types.ModuleType(module)
        setattr(stub, attribute, object)
        dependencies[module] = stub
    path = Path(__file__).resolve().parents[1] / 'src/dex_mimic/training/live_rotation.py'
    spec = importlib.util.spec_from_file_location('_live_index_metadata_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module.LiveRotation


LiveRotation = live_rotation_class()


class LiveGrowthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.seed = Path(self.tmp.name)
        manifest = self.seed / 'manifest.json'
        manifest.write_text(json.dumps(dict(complete=True, fixture_only=True)))
        self.path = self.seed / 'index.json'
        self.index = dict(seed_dataset_sha256=sha(manifest), revision=0,
                          frame_capacity=600000, records=[self.row(0), self.row(1)])
        self.save()
        # Only the metadata state required by the real validation method.
        self.rot = LiveRotation.__new__(LiveRotation)
        self.rot.index = dict(records=copy.deepcopy(self.index['records']),
                              frame_capacity=self.index['frame_capacity'])
        self.rot.dataset_sha256 = self.index['seed_dataset_sha256']
        self.rot.live_index = self.path
        self.rot.live_digest = None
        self.rot.live_revision = 0
        self.rot._read_live_index()

    def row(self, number):
        # These files identify metadata only; no tensor pack is loaded here.
        pack = self.seed / f'shard_{number}.bin'
        manifest = self.seed / f'shard_{number}.json'
        pack.write_bytes(f'fixture {number}'.encode())
        manifest.write_text(json.dumps(dict(fixture_only=True, shard=number)))
        return dict(pack=str(pack), manifest=str(manifest),
                    pack_sha256=sha(pack), manifest_sha256=sha(manifest))

    def save(self):
        self.path.write_text(json.dumps(self.index))

    def reject(self, message):
        previous = copy.deepcopy((self.rot.index, self.rot.live_digest, self.rot.live_revision))
        self.save()
        with self.assertRaisesRegex(ValueError, message):
            self.rot._read_live_index()
        self.assertEqual((self.rot.index, self.rot.live_digest, self.rot.live_revision), previous)

    def test_existing_pack_identity_is_immutable(self):
        self.index['records'][0]['pack_sha256'] = 'changed'
        self.reject('existing frozen shard')

    def test_index_cannot_remove_shards(self):
        self.index['records'].pop()
        self.reject('append-only')

    def test_resident_capacity_cannot_change(self):
        self.index['frame_capacity'] += 1
        self.reject('tensor capacity')

    def test_dataset_identity_cannot_change(self):
        self.index['seed_dataset_sha256'] = 'another dataset'
        self.reject('different seed release')

    def test_new_paths_must_be_absolute(self):
        row = self.row(2)
        row['pack'] = 'relative.bin'
        self.index['records'].append(row)
        self.reject('paths must be absolute')

    def test_valid_append_and_unchanged_read(self):
        self.index['records'].append(self.row(2))
        self.index['revision'] = 1
        self.save()
        self.assertTrue(self.rot._read_live_index())
        self.assertEqual(self.rot.live_revision, 1)
        self.assertEqual(self.rot.index['records'], self.index['records'])
        self.assertFalse(self.rot._read_live_index())


if __name__ == '__main__':
    unittest.main()
