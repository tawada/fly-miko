#!/usr/bin/env python3
"""Create a version-pinned output-neuron registry; does not simulate a brain."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "8587524c1748ce5ef2080822a2fc890fc03bf597"
SHA256 = "9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be"
URL = (
    f"https://raw.githubusercontent.com/flyconnectome/flywire_annotations/{REVISION}/"
    "supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)


def main():
    directory = ROOT / "data/brain"
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "flywire-neuron-annotations.tsv"
    if not source.exists():
        raise FileNotFoundError(
            "No local neuron annotations. This registry-only command does not download data because redistribution "
            "terms for this pinned TSV are unconfirmed. See docs/data-preparation.md; "
            "the repository does not include these annotations or derived bindings. To obtain the pinned publisher files explicitly, run scripts/prepare_reproduction.py --download.")
    if hashlib.sha256(source.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Local annotation checksum mismatch; use the pinned source")
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    descending = [row for row in rows if row["super_class"] == "descending"]
    # Exclude both segmentation and biological outlier flags, explicitly.
    selected = sorted(
        (row for row in descending if not row["status"]), key=lambda row: int(row["root_id"])
    )
    neurons = [
        {"id": row["root_id"], "cellType": row["cell_type"], "side": row["side"],
         "predictedTransmitter": row["top_nt"]}
        for row in selected
    ]
    ids = [row["id"] for row in neurons]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("Invalid neuron identifiers")
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    binding = {
        "id": f"flywire-783-descending-{digest}",
        "dataset": "flywire-783",
        "source": {"url": URL, "revision": REVISION, "sha256": SHA256},
        "selection": {"super_class": "descending", "status": "",
                      "totalDescending": len(descending),
                      "excludedFlagged": len(descending) - len(selected)},
        "neurons": neurons,
        "feedbackBinding": None,
        "note": "Output-neuron registry only. No connectome, simulator, or trained policy included.",
    }
    destination = directory / "flywire-783-binding.json"
    destination.write_text(json.dumps(binding, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(neurons)} output neurons, {len(descending) - len(selected)} flagged entries excluded")
    print(destination)
    ascending = [row for row in rows if row["super_class"] == "ascending"]
    inputs = sorted((row for row in ascending if not row["status"]), key=lambda row: int(row["root_id"]))
    input_neurons = [
        {"id": row["root_id"], "cellType": row["cell_type"], "side": row["side"]}
        for row in inputs
    ]
    input_digest = hashlib.sha256("\n".join(n["id"] for n in input_neurons).encode()).hexdigest()
    input_binding = {
        "id": f"flywire-783-ascending-{input_digest}",
        "dataset": "flywire-783", "role": "ascending-input-candidates",
        "source": binding["source"],
        "selection": {"super_class": "ascending", "status": "",
                      "totalAscending": len(ascending), "excludedFlagged": len(ascending) - len(inputs)},
        "neurons": input_neurons,
        "note": "Candidate interface only; these IDs are not identified human-joint sensory homologues.",
    }
    input_destination = directory / "flywire-783-input-binding.json"
    input_destination.write_text(json.dumps(input_binding, ensure_ascii=False, indent=2) + "\n")
    # Keep the output registry's ID/order unchanged while describing the new input boundary.
    binding["feedbackBinding"] = {
        "id": input_binding["id"], "file": input_destination.name,
        "status": "experimental-external-stimulation",
    }
    destination.write_text(json.dumps(binding, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(input_neurons)} ascending input candidates")
    print(input_destination)


if __name__ == "__main__":
    main()
