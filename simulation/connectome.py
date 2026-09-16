"""Download and compile the version-pinned Shiu et al. FlyWire 783 connectivity."""
import csv
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import pyarrow.parquet as pq

from .biped import ROOT

REVISION = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
SOURCE = f"https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{REVISION}"
FILES = {
    "Connectivity_783.parquet": "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347",
    "Completeness_783.csv": "bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311",
    "LICENSE": "3621f6d6476189190e2960fa43f11b275ba4eca848fdac50a1ba2925de6223e8",
}
COMPILED = ROOT / "data/brain/compiled"


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compile_connectome():
    directory = ROOT / "data/brain/upstream"
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = directory / name
        if not path.exists():
            print(f"Downloading {name}", flush=True)
            temp = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(f"{SOURCE}/{name}", temp)
            if file_hash(temp) != expected:
                raise ValueError(f"Downloaded checksum mismatch: {name}")
            temp.replace(path)
        if file_hash(path) != expected:
            raise ValueError(f"Source checksum mismatch: {name}")
    with (directory / "Completeness_783.csv").open() as stream:
        reader = csv.reader(stream)
        next(reader)
        ids = [row[0] for row in reader]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate completeness IDs")
    numeric_ids = np.array(ids, dtype=np.int64)
    neuron_count = len(ids)
    source = pq.ParquetFile(directory / "Connectivity_783.parquet")
    edge_count = source.metadata.num_rows
    COMPILED.mkdir(parents=True, exist_ok=True)
    post = np.lib.format.open_memmap(COMPILED / "posts.npy", mode="w+", dtype=np.int32, shape=(edge_count,))
    weights = np.lib.format.open_memmap(COMPILED / "weights.npy", mode="w+", dtype=np.float32, shape=(edge_count,))
    degree = np.zeros(neuron_count, dtype=np.int64)
    cursor, previous_pre, inhibitory = 0, -1, 0
    for batch in source.iter_batches(batch_size=500_000, columns=[
        "Presynaptic_ID", "Postsynaptic_ID", "Presynaptic_Index",
        "Postsynaptic_Index", "Excitatory x Connectivity",
    ]):
        columns = {name: batch.column(name).to_numpy() for name in batch.schema.names}
        pres = columns["Presynaptic_Index"]
        posts = columns["Postsynaptic_Index"]
        raw_weights = columns["Excitatory x Connectivity"]
        if pres.min() < 0 or posts.min() < 0 or max(pres.max(), posts.max()) >= neuron_count:
            raise ValueError("Connectivity index outside completeness table")
        if pres[0] < previous_pre or np.any(pres[1:] < pres[:-1]):
            raise ValueError("Source must be sorted by presynaptic index")
        if not np.array_equal(numeric_ids[pres], columns["Presynaptic_ID"]):
            raise ValueError("Presynaptic ID/index mismatch")
        if not np.array_equal(numeric_ids[posts], columns["Postsynaptic_ID"]):
            raise ValueError("Postsynaptic ID/index mismatch")
        end = cursor + len(pres)
        post[cursor:end] = posts
        weights[cursor:end] = raw_weights * 0.275  # mV per anatomical synapse, upstream parameter
        degree += np.bincount(pres, minlength=neuron_count)
        inhibitory += int(np.count_nonzero(raw_weights < 0))
        previous_pre = int(pres[-1])
        cursor = end
    if cursor != edge_count:
        raise ValueError("Incomplete connectivity read")
    post.flush()
    weights.flush()
    np.save(COMPILED / "indptr.npy", np.concatenate(([0], np.cumsum(degree))))
    (COMPILED / "neuron-ids.json").write_text(json.dumps(ids))
    known = set(ids)
    reconciliation = {}
    bindings = {}
    for filename, output in [
        ("flywire-783-binding.json", "output-binding.json"),
        ("flywire-783-input-binding.json", "input-binding.json"),
    ]:
        binding = json.loads((ROOT / "data/brain" / filename).read_text())
        original_id = binding["id"]
        excluded = [n["id"] for n in binding["neurons"] if n["id"] not in known]
        binding["neurons"] = [n for n in binding["neurons"] if n["id"] in known]
        digest = hashlib.sha256("\n".join(n["id"] for n in binding["neurons"]).encode()).hexdigest()
        binding["id"] = f"flywire-783-{output.removesuffix('-binding.json')}-{digest}"
        binding["parentBindingId"] = original_id
        binding["excludedWithoutConnectivity"] = excluded
        binding["connectomeRevision"] = REVISION
        bindings[output] = binding
        reconciliation[output] = {"count": len(binding["neurons"]), "excludedIds": excluded}
    bindings["output-binding.json"]["feedbackBinding"] = {
        "id": bindings["input-binding.json"]["id"], "file": "input-binding.json",
        "status": "connected-experimental-stimulation",
    }
    for filename, binding in bindings.items():
        (COMPILED / filename).write_text(json.dumps(binding, indent=2))
    hashes = {name: file_hash(COMPILED / name) for name in [
        "indptr.npy", "posts.npy", "weights.npy", "neuron-ids.json",
        "input-binding.json", "output-binding.json",
    ]}
    manifest = {
        "id": f"shiu-flywire-783-{REVISION}",
        "revision": REVISION, "dataset": "flywire-783", "source": SOURCE,
        "sourceChecksums": FILES, "compiledChecksums": hashes,
        "neuronCount": neuron_count, "connectionCount": edge_count,
        "negativeConnections": inhibitory, "weightUnits": "mV", "bindings": reconciliation,
        "model": "leaky-integrate-and-fire", "implementation": "fly-miko-numba-v1",
        "note": "Full supplied connectivity; independent numerical implementation, not a validated Brian2 reproduction.",
    }
    (COMPILED / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ["neuronCount", "connectionCount", "bindings"]}, indent=2))
    return manifest


class Connectome:
    def __init__(self, directory=COMPILED, verify=True):
        self.directory = Path(directory)
        if not (self.directory / "manifest.json").exists():
            raise FileNotFoundError("Run npm run brain:build first")
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if verify:
            for name, expected in self.manifest["compiledChecksums"].items():
                if file_hash(self.directory / name) != expected:
                    raise ValueError(f"Compiled graph checksum mismatch: {name}")
        self.ids = json.loads((self.directory / "neuron-ids.json").read_text())
        self.index = {identifier: i for i, identifier in enumerate(self.ids)}
        self.indptr = np.load(self.directory / "indptr.npy", mmap_mode="r")
        self.posts = np.load(self.directory / "posts.npy", mmap_mode="r")
        self.weights = np.load(self.directory / "weights.npy", mmap_mode="r")
        self.input_binding = json.loads((self.directory / "input-binding.json").read_text())
        self.output_binding = json.loads((self.directory / "output-binding.json").read_text())
        self.input_indices = np.array([self.index[n["id"]] for n in self.input_binding["neurons"]], dtype=np.int32)
        self.output_indices = np.array([self.index[n["id"]] for n in self.output_binding["neurons"]], dtype=np.int32)


if __name__ == "__main__":
    compile_connectome()
