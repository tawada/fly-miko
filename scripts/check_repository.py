"""Check source candidates, including force-added local artifacts, before sharing."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BLOCKED = {'assets', 'downloads', 'data', 'runs', 'renders', 'checkpoints', 'node_modules', 'dist', '__pycache__'}
EXTENSIONS = {'.pmx', '.pmd', '.vrm', '.fbx', '.glb', '.gltf', '.blend', '.zip', '.gz', '.tar', '.7z', '.npz', '.npy', '.parquet', '.feather', '.ply', '.swc', '.obj', '.stl', '.mp4', '.webm', '.mov', '.pyc', '.nbc', '.nbi', '.ttf', '.otf', '.png', '.jpg', '.jpeg', '.webp', '.gif', '.tga', '.bmp', '.csv', '.tsv', '.pkl', '.pickle', '.bin', '.onnx', '.h5', '.hdf5'}


def violation(name, root):
    path = Path(name)
    if any(p in BLOCKED or p.startswith('.venv') for p in path.parts):
        return 'local artifact directory'
    if path.suffix.lower() in EXTENSIONS:
        return 'binary/model/data artifact'
    if path.name.startswith('.env') and path.name != '.env.example':
        return 'local credentials'
    if any(p.is_symlink() for p in [root / path, *(root / path).parents] if p != root.parent):
        return 'symlink'
    return None


def main():
    paths = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0')
    errors = [f'{p}: {reason}' for p in set(paths) if p and (reason := violation(p, ROOT))]
    for p in ['NOTICE.md', 'LICENSES.md', 'third_party/Drosophila_brain_model.LICENSE']:
        if not (ROOT / p).is_file(): errors.append(f'Missing {p}')
    for record in json.loads((ROOT / 'third_party/license-sources.json').read_text())['licenses']:
        path = ROOT / 'third_party' / record['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            errors.append(f'License checksum mismatch: {path}')
    package = json.loads((ROOT / 'package.json').read_text())
    if package.get('private') is not True or package.get('license') != 'UNLICENSED':
        errors.append('Original-code licensing status changed; review LICENSES.md')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    print(f'Repository candidates checked: {len(set(paths) - {""})}; local artifacts excluded; notices intact.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
