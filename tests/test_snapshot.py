import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from rescuehandsai.snapshot import (SnapshotRefused, asset_files, asset_hash, file_digest, runtime_versions,
                                    snapshot_record, source_hash, uncommitted_files)


def git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   check=True, capture_output=True)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "src").mkdir()
        (self.root / "configs").mkdir()
        (self.root / "src" / "a.py").write_text("x = 1\n")
        (self.root / "configs" / "c.json").write_text("{}\n")
        assets = self.root / "asset" / "meshes"
        assets.mkdir(parents=True)
        (assets / "part.stl").write_bytes(b"\x00\x01binary")
        self.xml = self.root / "asset" / "robot.xml"
        self.xml.write_text('<mujoco><compiler meshdir="meshes"/><asset><mesh file="part.stl"/></asset></mujoco>\n')
        git(self.root, "init", "-q")
        git(self.root, "add", ".")
        git(self.root, "commit", "-q", "-m", "init")

    def tearDown(self):
        self.tmp.cleanup()

    def test_text_hash_ignores_crlf_but_binary_hash_does_not(self):
        a, b = self.root / "x.py", self.root / "y.py"
        a.write_bytes(b"a = 1\n")
        b.write_bytes(b"a = 1\r\n")
        self.assertEqual(file_digest(a), file_digest(b))
        c, d = self.root / "x.stl", self.root / "y.stl"
        c.write_bytes(b"\n")
        d.write_bytes(b"\r\n")
        self.assertNotEqual(file_digest(c), file_digest(d))

    def test_untracked_source_changes_hash_and_blocks_strict_runs(self):
        clean_digest, _ = source_hash(self.root)
        self.assertEqual(snapshot_record(self.root, self.xml, strict=True)["source_sha256"], clean_digest)
        (self.root / "src" / "extra.py").write_text("y = 2\n")
        self.assertEqual(uncommitted_files(self.root), ["src/extra.py"])
        self.assertNotEqual(source_hash(self.root)[0], clean_digest)
        with self.assertRaises(SnapshotRefused):
            snapshot_record(self.root, self.xml, strict=True)
        record = snapshot_record(self.root, self.xml, strict=False)
        self.assertIn("src/extra.py", record["uncommitted"])
        self.assertEqual(record["uncommitted"]["src/extra.py"],
                          file_digest(self.root / "src" / "extra.py"))

    def test_renamed_file_reports_destination_path(self):
        git(self.root, "mv", "src/a.py", "src/b.py")
        self.assertIn("src/b.py", uncommitted_files(self.root))
        self.assertNotIn("src/a.py -> src/b.py", uncommitted_files(self.root))
        record = snapshot_record(self.root, self.xml, strict=False)
        self.assertEqual(record["uncommitted"]["src/b.py"],
                          file_digest(self.root / "src" / "b.py"))

    def test_pycache_is_ignored(self):
        before = source_hash(self.root)[0]
        cache = self.root / "src" / "__pycache__"
        cache.mkdir()
        (cache / "a.cpython-312.pyc").write_bytes(b"junk")
        self.assertEqual(source_hash(self.root)[0], before)

    def test_asset_hash_covers_referenced_meshes(self):
        self.assertEqual([p.name for p in asset_files(self.xml)], ["robot.xml", "part.stl"])
        before = asset_hash(self.xml)[0]
        (self.root / "asset" / "meshes" / "part.stl").write_bytes(b"changed")
        self.assertNotEqual(asset_hash(self.xml)[0], before)
        (self.root / "asset" / "meshes" / "part.stl").unlink()
        with self.assertRaises(FileNotFoundError):
            asset_files(self.xml)

    def test_model_and_noise_files_and_runtime_are_recorded(self):
        model = self.root / "model.bin"
        model.write_bytes(b"weights")
        noise = self.root / "noise.npy"
        noise.write_bytes(b"noise")
        record = snapshot_record(self.root, self.xml, strict=True, model_files={"export": model}, noise_file=noise)
        self.assertEqual(record["model_files"]["export"], file_digest(model))
        self.assertEqual(record["noise_sha256"], file_digest(noise))
        runtime = runtime_versions()
        self.assertIn("python", runtime)
        self.assertIn("mujoco", runtime["packages"])
        self.assertIn("devices", runtime)

    def test_broken_openvino_install_does_not_crash_runtime_versions(self):
        # A present-but-broken OpenVINO install (missing plugin DLLs, partial install)
        # must degrade like an absent one, not abort runtime_versions()/snapshot_record.
        fake_openvino = types.ModuleType("openvino")

        class BrokenCore:
            def __init__(self):
                raise RuntimeError("missing plugin dll")

        fake_openvino.Core = BrokenCore
        with mock.patch.dict(sys.modules, {"openvino": fake_openvino}):
            runtime = runtime_versions()
        self.assertIn("openvino", runtime["devices"])
        self.assertIn("error", runtime["devices"]["openvino"])
        self.assertIn("RuntimeError", runtime["devices"]["openvino"]["error"])


if __name__ == "__main__":
    unittest.main()
