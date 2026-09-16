"""Train the connectome-to-body readout with resumable population search."""
import argparse
from datetime import datetime, timezone
import json
import math
import re

import numpy as np

from .biped import ROOT
from .storage import atomic_output, check_output_directory

RUNS = ROOT / "runs/training"


def atomic_json(path, value):
    content = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    with atomic_output(path) as stream:
        stream.write(content.encode("utf-8"))


def checkpoint(path, **values):
    with atomic_output(path) as stream:
        np.savez_compressed(stream, **values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="random-head-bias-02")
    parser.add_argument("--algorithm", choices=["mutation", "retained"], default="mutation",
                        help="mutation: elite + 7 mutations + 2 random; retained is a compatibility alias")
    parser.add_argument("--init-from", help="Initialize a new retained run from an old latest.npz best vector")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-storage", action="store_true",
                        help="Check output storage without loading the brain or running training")
    parser.add_argument("--task", choices=["head-height"], default="head-height")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--population", type=int, choices=[10], default=10)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=10.)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--neural-dt-ms", type=float, default=0.1)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 10000 or not 4 <= args.population <= 128 or not 1 <= args.workers <= 16:
        parser.error("Invalid iteration/population/worker range")
    if not math.isfinite(args.seconds) or not 0.1 <= args.seconds <= 120 or args.seed < 0:
        parser.error("Invalid episode duration/seed")
    if args.check_storage:
        if not args.name or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.name):
            parser.error("--check-storage requires a valid --name")
        check_output_directory(RUNS)
        check_output_directory(RUNS / args.name)
        print(f"Storage check passed: {RUNS / args.name}")
        return
    if args.algorithm in ("retained", "mutation"):
        from .population_search import train as train_population
        if args.population != 10 or args.task != "head-height":
            parser.error("Retained scenario requires --population 10 --task head-height")
        from .generation_archive import background_archive
        if not args.name:
            args.name = datetime.now(timezone.utc).strftime("random-%Y%m%d-%H%M%S")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.name):
            parser.error("Invalid run name")
        with background_archive(RUNS / args.name):
            train_population(args)



if __name__ == "__main__":
    main()
