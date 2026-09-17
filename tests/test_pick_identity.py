import copy
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

from rescuehandsai.pick_identity import (block_of, check_seed_blocks, config_hash, find_duplicates,
                                         load_seed_blocks, scene_hash, settings_hash, settings_record, text_digest)
from rescuehandsai.scene import ROOT, load_config, sample_params


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.asset = ROOT / ".cache/menagerie/robotstudio_so101/so101.xml"

    def test_line_endings_do_not_change_text_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            lf, crlf = Path(tmp) / "a.json", Path(tmp) / "b.json"
            lf.write_bytes(b'{"a": 1}\n{"b": 2}\n')
            crlf.write_bytes(b'{"a": 1}\r\n{"b": 2}\r\n')
            self.assertEqual(text_digest(lf), text_digest(crlf))

    def test_config_hash_depends_on_physics_version(self):
        self.assertEqual(config_hash(2, self.asset), config_hash(2, self.asset))
        self.assertNotEqual(config_hash(1, self.asset), config_hash(2, self.asset))

    def test_scene_hash_depends_on_version_and_scene(self):
        a, b = sample_params(self.config, 1), sample_params(self.config, 2)
        self.assertNotEqual(scene_hash(a, self.config, 1), scene_hash(a, self.config, 2))
        self.assertNotEqual(scene_hash(a, self.config, 2), scene_hash(b, self.config, 2))

    def test_settings_hash_ignores_seed_and_float_noise(self):
        params = sample_params(self.config, 3)
        record = settings_record(params)
        self.assertIsInstance(record["poses"]["fork"], list)
        self.assertEqual(settings_hash(params), settings_hash(replace(params, seed=999)))
        nudged = replace(params, light_diffuse=params.light_diffuse + 1e-9)
        self.assertEqual(settings_hash(params), settings_hash(nudged))
        self.assertNotEqual(settings_hash(params), settings_hash(sample_params(self.config, 4)))

    def test_duplicates_across_lists_are_found(self):
        a, b = sample_params(self.config, 1), sample_params(self.config, 2)
        copy_of_a = replace(a, seed=3200000)
        found = find_duplicates({"train": [a, b], "test_main": [copy_of_a]})
        self.assertEqual(len(found), 1)
        self.assertEqual(sorted(found[0]["members"]), [["test_main", 3200000], ["train", 1]])
        self.assertEqual(find_duplicates({"train": [a], "dev": [b]}), [])

    def test_seed_blocks_do_not_overlap_and_legacy_is_reported_incomplete(self):
        report = check_seed_blocks(load_seed_blocks())
        self.assertEqual(report["overlaps"], [])
        self.assertFalse(report["legacy_provenance_complete"])
        self.assertEqual(block_of(3200050, load_seed_blocks()), "test_main")
        self.assertIsNone(block_of(42, load_seed_blocks()))

    def test_overlapping_blocks_are_reported(self):
        config = copy.deepcopy(load_seed_blocks())
        config["blocks"]["train"] = [150000, 160000]
        config["blocks"]["dev"] = [3099999, 3200001]
        overlaps = {tuple(sorted(o)) for o in check_seed_blocks(config)["overlaps"]}
        self.assertIn(("legacy_0", "train"), overlaps)
        self.assertIn(("dev", "test_main"), overlaps)


if __name__ == "__main__":
    unittest.main()
