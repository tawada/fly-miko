import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_reproduction', ROOT / 'scripts/prepare_reproduction.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class ReproductionTests(unittest.TestCase):
    def test_download_verifies_before_publishing_and_never_overwrites_bad_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            publisher = root / 'publisher.txt'; publisher.write_bytes(b'expected data')
            source = {'path': 'data/source.tsv', 'url': publisher.as_uri(),
                      'sha256': hashlib.sha256(publisher.read_bytes()).hexdigest()}
            with self.assertRaises(FileNotFoundError): prepare.fetch(source, root)
            prepare.fetch(source, root, download=True)
            self.assertEqual((root / source['path']).read_bytes(), b'expected data')
            # Reuse works without a network/publisher file.
            publisher.unlink()
            prepare.fetch(source, root)
            (root / source['path']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError): prepare.fetch(source, root, download=True)
            self.assertEqual((root / source['path']).read_bytes(), b'corrupt')
            (root / source['path']).unlink()
            publisher.write_bytes(b'wrong download')
            with self.assertRaises(ValueError): prepare.fetch(source, root, download=True)
            self.assertFalse((root / source['path']).exists())
            self.assertEqual(list((root / 'data').glob('*.tmp')), [])

    def test_initial_readout_reconstructs_exact_parameters_without_training_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / 'config').mkdir()
            for name in ['reproduction.json', 'reproduction-initial.json']:
                shutil.copy2(ROOT / 'config' / name, root / 'config' / name)
            specification = json.loads((root / 'config/reproduction.json').read_text())
            initial = json.loads((root / specification['initialFile']).read_text())
            checkpoint = prepare.prepare_initial(root, specification)
            with np.load(checkpoint, allow_pickle=False) as data:
                np.testing.assert_array_equal(data['best'], initial['parameters'])
                self.assertEqual(data['best'].shape, (594,))
            self.assertEqual(prepare.prepare_initial(root, specification), checkpoint)
            self.assertFalse((root / 'runs').exists())
            with checkpoint.open('wb') as output: np.savez(output, best=np.ones(594))
            with self.assertRaises(ValueError): prepare.prepare_initial(root, specification)

    def test_published_initial_file_and_parameter_hashes(self):
        specification = json.loads((ROOT / 'config/reproduction.json').read_text())
        initial_path = ROOT / specification['initialFile']
        self.assertEqual(prepare.file_hash(initial_path), specification['initialFileSha256'])
        initial = json.loads(initial_path.read_text())
        parameters = np.asarray(initial['parameters'], dtype='<f8')
        self.assertEqual(hashlib.sha256(parameters.tobytes()).hexdigest(), specification['initialParameterSha256'])
