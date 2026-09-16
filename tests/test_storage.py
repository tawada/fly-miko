import errno
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from simulation.storage import atomic_output, check_output_directory
from simulation.train import atomic_json, checkpoint


class StorageTests(unittest.TestCase):
    def test_transient_permission_denied_retries_without_removing_old_file(self):
        import os
        real_replace = os.replace
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "progress.json"
            target.write_text('{"generation":1}')
            calls = []
            def busy(source, destination):
                calls.append(source)
                if len(calls) < 3:
                    self.assertEqual(json.loads(target.read_text()), {"generation": 1})
                    raise PermissionError(errno.EACCES, "busy destination")
                real_replace(source, destination)
            with patch("simulation.storage.os.replace", side_effect=busy), patch("simulation.storage.time.sleep"):
                atomic_json(target, {"generation": 2})
            self.assertEqual(len(calls), 3)
            self.assertEqual(json.loads(target.read_text()), {"generation": 2})
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_permanent_denial_keeps_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "latest.npz"
            checkpoint(target, generation=1, best=np.array([1.]))
            before = target.read_bytes()
            with patch("simulation.storage.os.replace", side_effect=PermissionError(errno.EACCES, "denied")), \
                    patch("simulation.storage.time.sleep") as sleep:
                with self.assertRaises(PermissionError):
                    checkpoint(target, generation=2, best=np.array([2.]))
            self.assertEqual(sleep.call_count, 9)
            self.assertEqual(target.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_unrelated_io_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("simulation.storage.os.replace", side_effect=OSError(errno.ENOSPC, "full")), \
                    patch("simulation.storage.time.sleep") as sleep:
                with self.assertRaises(OSError):
                    atomic_json(Path(directory) / "progress.json", {})
            sleep.assert_not_called()

    def test_unique_temporaries_ignore_stale_shared_tmp_name(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "progress.json"
            stale = target.with_suffix(".json.tmp")
            stale.write_text("stale")
            with atomic_output(target) as first:
                first.write(b"first")
                with atomic_output(target) as second:
                    second.write(b"second")
                self.assertEqual(target.read_bytes(), b"second")
            self.assertEqual(target.read_bytes(), b"first")
            self.assertEqual(stale.read_text(), "stale")

    def test_probe_does_not_modify_saved_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "progress.json"
            target.write_text("original")
            check_output_directory(directory)
            self.assertEqual(target.read_text(), "original")
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_writer_exception_preserves_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "latest.npz"
            target.write_bytes(b"original")
            with self.assertRaises(RuntimeError):
                with atomic_output(target) as stream:
                    stream.write(b"partial")
                    raise RuntimeError("interrupted serialization")
            self.assertEqual(target.read_bytes(), b"original")
            self.assertEqual(list(Path(directory).iterdir()), [target])


if __name__ == "__main__":
    unittest.main()
