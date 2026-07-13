from __future__ import annotations

import json
import os
import subprocess
from typing import List, Sequence, Tuple

import numpy as np

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

def write_affinity_yaml(yaml_path: str, heavy_chain: str, light_chain: str, ligand_smiles: str) -> None:
    with open(yaml_path, "w") as f:
        f.write("sequences:\n")
        f.write("    - protein:\n")
        f.write("        id: H\n")
        f.write(f"        sequence: {heavy_chain}\n")
        f.write("    - protein:\n")
        f.write("        id: L\n")
        f.write(f"        sequence: {light_chain}\n")
        f.write("    - ligand:\n")
        f.write("        id: d\n")
        f.write(f"        smiles: '{ligand_smiles}'\n")
        f.write("\n")
        f.write("properties:\n")
        f.write("    - affinity:\n")
        f.write("        binder: d\n")


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
# Batch Boltz-2 invocation -- same subprocess/env pattern as the user's
# run_boltz, pointed at a directory instead of a single yaml file (the
# `boltz predict` CLI loads the model once and runs every .yaml inside).
# ============================================================

def run_boltz_batch(
    batch_dir: str,
    out_dir: str,
    max_parallel_samples: int = 1,
    diffusion_samples: int = 1,
    use_msa_server: bool = True,
    accelerator: str = "gpu",
) -> bool:
    cmd = [
        "conda", "run", "-n", "boltz2",
        "boltz", "predict", batch_dir,
        "--max_parallel_samples", str(max_parallel_samples),
        "--diffusion_samples", str(diffusion_samples),
        "--use_potentials",
        "--out_dir", out_dir,
        "--accelerator", accelerator,
    ]
    if use_msa_server:
        cmd.append("--use_msa_server")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        "/usr/local/envs/boltz2/lib/python3.10/site-packages/nvidia/cu13/lib:"
        "/usr/local/envs/boltz2/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:"
        "/usr/local/envs/boltz2/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:"
        "/usr/lib64-nvidia:"
        + env.get("LD_LIBRARY_PATH", "")
    )

    try:
        result = subprocess.run(cmd, env=env, text=True, capture_output=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            print(f"Boltz batch failed with exit code {result.returncode}")
            return False
        print("Batch prediction completed successfully.\nFiles saved at:", out_dir)
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


# ============================================================
# Result parsing + reward
# ============================================================

def read_affinity_json(out_dir: str, unique_id: str) -> dict:
    path = os.path.join(out_dir, "predictions", unique_id, f"affinity_{unique_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No affinity output for candidate '{unique_id}' at {path} -- "
            f"the Boltz batch call may have failed or silently skipped this candidate."
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
    Callable matching pool_search's score_fn(full_seqs) -> np.ndarray contract.
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

    def __call__(self, full_seqs: List[str]) -> np.ndarray:
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
        for i, unique_id in enumerate(unique_ids):
            data = read_affinity_json(out_dir, unique_id)
            scores[i] = compute_log_reward(data["affinity_probability_binary"], data["affinity_pred_value"])
        return scores
