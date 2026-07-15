from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from typing import List, Sequence, Tuple

import numpy as np
import yaml

# ============================================================
# Fixed antibody context: everything except the 7 designable
# positions inside CDRH3 that pool_search.py actually mutates.
# CDRH3 here must match pool_search.SearchConfig.root_sequence exactly.
# ============================================================

LIGHT_CHAIN = (
    "DILMTQTPSSMSVSLGDTVTITCHASRGVRSYIGWMQQKPGKSFKGLIYNGTNLEDGISSRFGGRGSGT"
    "DYSLTISGLESEDFGDYYCVQYAQFPYTFGGGTKLEVK"
)
HEAVY_BEFORE_CDRH3 = (
    "EVPLVESGGGLVKPGGSLRLSCTASGFPFSRSAMSWVRQSPDKRLEWVAEISSGGFFTSYVDTVTGRFTIS"
    "RDNAKNILYLEMSSLRPGDTAIYYC"
)
HEAVY_AFTER_CDRH3 = "WGQGTSVTVSS"
LIGAND_SMILES = "C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=C3C=CC(=C4)O"  # estradiol

ROOT_CDRH3 = "ARERGIHYYGSSEIQDY"  # sanity-check anchor; must match pool_search's root_sequence


def assemble_full_heavy_chain(cdrh3: str) -> str:
    return HEAVY_BEFORE_CDRH3 + cdrh3 + HEAVY_AFTER_CDRH3


# ============================================================
# Batch YAML writing -- adapted from the user's create_affinity_yaml,
# one uniquely-named file per candidate instead of one file total.
# ============================================================

def write_affinity_yaml(yaml_path: str, heavy_chain: str, light_chain: str, ligand_smiles: str,
                        ligand_id: str = "d") -> None:
    # Matches the validated Colab notebook exactly: `version: 1` header (Boltz-2
    # schema field the hand-written writer omitted) and yaml.safe_dump so odd
    # characters in a SMILES/sequence can't corrupt the file.
    yaml_data = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "H", "sequence": heavy_chain}},
            {"protein": {"id": "L", "sequence": light_chain}},
            {"ligand": {"id": ligand_id, "smiles": ligand_smiles}},
        ],
        "properties": [
            {"affinity": {"binder": ligand_id}},
        ],
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False, default_flow_style=False)


def write_batch_yamls(batch_dir: str, candidates: Sequence[Tuple[str, str]]) -> None:
    """candidates: (unique_id, cdrh3) pairs. Writes one YAML per candidate
    into batch_dir, named <unique_id>.yaml. unique_id must never repeat
    across the whole search run -- boltz reuses cached predictions for a
    filename it's already seen in that output directory, which would
    silently return a stale result for a different sequence otherwise."""
    os.makedirs(batch_dir, exist_ok=True)
    for unique_id, cdrh3 in candidates:
        full_heavy = assemble_full_heavy_chain(cdrh3)
        yaml_path = os.path.join(batch_dir, f"{unique_id}.yaml")
        write_affinity_yaml(yaml_path, full_heavy, LIGHT_CHAIN, LIGAND_SMILES)


# ============================================================
# Batch Boltz-2 invocation. Points `boltz predict` at a directory (the CLI
# loads the model once and runs every .yaml inside). Intended to run FROM
# INSIDE the boltz environment (e.g. `conda run -n boltz2 python run_search.py`),
# so it calls `boltz` directly rather than wrapping each call in `conda run`.
# ============================================================

def _nvidia_lib_dirs() -> List[str]:
    """Every site-packages/nvidia/*/lib dir on sys.path's site-packages, so
    boltz's CUDA libs are found regardless of the exact CUDA version pip
    pulled -- replaces the old hardcoded, version-specific (and partly
    nonexistent) 'nvidia/cu13/lib' path list."""
    dirs: List[str] = []
    for base in {os.path.dirname(os.path.dirname(p)) for p in sys.path if p.endswith("site-packages")} \
            | {p for p in sys.path if p.endswith("site-packages")}:
        dirs.extend(glob.glob(os.path.join(base, "nvidia", "*", "lib")))
    return sorted(set(dirs))


def run_boltz_batch(
    batch_dir: str,
    out_dir: str,
    max_parallel_samples: int = 1,
    diffusion_samples: int = 1,
    diffusion_samples_affinity: int = 5,
    use_msa_server: bool = True,
    use_potentials: bool = True,
    accelerator: str = "gpu",
    devices: int = 1,
    num_workers: int = 2,
    preprocessing_threads: int = 8,
    override: bool = True,
) -> bool:
    cmd = [
        "boltz", "predict", str(batch_dir),
        "--out_dir", str(out_dir),
        "--accelerator", accelerator,
        "--devices", str(devices),
        "--diffusion_samples", str(diffusion_samples),
        "--max_parallel_samples", str(max_parallel_samples),
        "--diffusion_samples_affinity", str(diffusion_samples_affinity),
        "--num_workers", str(num_workers),
    ]
    if preprocessing_threads is not None:
        cmd += ["--preprocessing-threads", str(preprocessing_threads)]
    if use_msa_server:
        cmd.append("--use_msa_server")
    if use_potentials:
        cmd.append("--use_potentials")
    if override:
        cmd.append("--override")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(
        _nvidia_lib_dirs() + ["/usr/lib64-nvidia", env.get("LD_LIBRARY_PATH", "")]
    )

    try:
        # No capture_output: let Boltz progress stream live to the notebook,
        # which matters across a multi-hour search. stderr surfaces on failure.
        result = subprocess.run(cmd, env=env, text=True, check=False)
        if result.returncode != 0:
            print(f"Boltz batch failed with exit code {result.returncode}")
            return False
        return True
    except Exception as e:
        print(f"An error occurred launching Boltz: {e}")
        return False


# ============================================================
# Result parsing + reward
# ============================================================

def read_affinity_json(out_dir: str, unique_id: str) -> dict:
    # Boltz's exact output layout has varied across versions (e.g.
    # <out>/predictions/<id>/affinity_<id>.json vs a boltz_results_* wrapper
    # dir). Try the canonical path first, then fall back to a recursive glob
    # for affinity_<id>.json anywhere under out_dir, so a layout change surfaces
    # as "found elsewhere" rather than a spurious FileNotFoundError.
    canonical = os.path.join(out_dir, "predictions", unique_id, f"affinity_{unique_id}.json")
    path = canonical
    if not os.path.exists(path):
        matches = glob.glob(os.path.join(out_dir, "**", f"affinity_{unique_id}.json"), recursive=True)
        if matches:
            path = matches[0]
        else:
            raise FileNotFoundError(
                f"No affinity output for candidate '{unique_id}' under {out_dir} "
                f"(looked for affinity_{unique_id}.json) -- the Boltz batch call may "
                f"have failed or silently skipped this candidate."
            )
    with open(path) as f:
        return json.load(f)


def compute_log_reward(affinity_probability_binary: float, affinity_pred_value: float, eps: float = 1e-6) -> float:
    """
    S(x) as stored/ranked throughout pool_search:
        log(reward), where reward = affinity_probability_binary * 10^(-2*affinity_pred_value)
    Linear in affinity_pred_value (constant sensitivity across its range,
    unlike the raw exponential reward -- see prior discussion). Use
    score_sign=+1 in SearchConfig with this (higher = better). Raw reward
    is recoverable via exp(this value) if ever needed for reporting.
    """
    p = max(affinity_probability_binary, eps)
    return float(np.log(p) - 2 * np.log(10) * affinity_pred_value)


# ============================================================
# score_fn factory for pool_search.run_search
# ============================================================

class Boltz2ScoreFn:
    """
    Callable matching pool_search's score_fn contract. Returns a tuple
    (scores, extras): `scores` is the np.ndarray of combined S(x) log-rewards
    the search ranks on, and `extras` is a per-candidate list of dicts with the
    raw Boltz-2 components {"affinity_value", "affinity_probability"} behind
    each score. pool_search.run_search accepts this tuple form and uses the
    extras to populate the writable score log; it also still accepts a plain
    ndarray, so returning the raw components is optional from its side.

    full_seqs are the assembled 17-residue CDRH3 strings pool_search produces
    internally (config.root_sequence's length) -- NOT the whole antibody.
    This wraps each one into the full heavy chain + light chain + ligand
    context before calling Boltz-2 as one batch.

    Maintains a monotonically increasing counter for unique YAML filenames
    across the WHOLE search run (not just one batch), so no filename is ever
    reused -- avoids boltz's cache-reuse-by-filename behavior silently
    returning a stale prediction for a different sequence.
    """

    def __init__(self, work_dir: str, **boltz_kwargs):
        self.work_dir = work_dir
        self.boltz_kwargs = boltz_kwargs
        self._counter = 0

    def __call__(self, full_seqs: List[str]) -> Tuple[np.ndarray, List[dict]]:
        start = self._counter
        self._counter += len(full_seqs)

        batch_id = f"batch_{start:06d}"
        unique_ids = [f"cand_{start + i:08d}" for i in range(len(full_seqs))]

        batch_dir = os.path.join(self.work_dir, batch_id, "yamls")
        out_dir = os.path.join(self.work_dir, batch_id, "out")

        write_batch_yamls(batch_dir, list(zip(unique_ids, full_seqs)))

        ok = run_boltz_batch(batch_dir, out_dir, **self.boltz_kwargs)
        if not ok:
            raise RuntimeError(f"Boltz batch prediction failed for {batch_id} (see stderr above)")

        scores = np.zeros(len(full_seqs))
        extras: List[dict] = []
        for i, unique_id in enumerate(unique_ids):
            data = read_affinity_json(out_dir, unique_id)
            v = float(data["affinity_pred_value"])
            p = float(data["affinity_probability_binary"])
            scores[i] = compute_log_reward(p, v)
            extras.append({"affinity_value": v, "affinity_probability": p})
        return scores, extras
