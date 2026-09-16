import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('audit', Path(__file__).resolve().parents[1] / 'scripts/check_repository.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class RepositoryBoundary(unittest.TestCase):
    def test_local_artifacts_outside_normal_directories(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ['demo/model.pmx', 'docs/result.npz', 'data/coordinates.json', '.venv-other/bin/python', '.env.secret']:
                self.assertIsNotNone(audit.violation(name, root), name)
            self.assertIsNone(audit.violation('third_party/licenses/source-LICENSE.txt', root))

    def test_symlink_cannot_disguise_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'hidden.pmx').write_text('model')
            (root / 'example.txt').symlink_to(root / 'hidden.pmx')
            self.assertEqual(audit.violation('example.txt', root), 'symlink')
