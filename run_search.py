"""
Entry point for the Boltz-2 pool search. Wires together:
  - SearchConfig / run_search        (pool_search.py)
  - BitmapSafetyChecker + facts      (build_universe_bitmap.py, precomputed .bin)
  - Boltz2ScoreFn                    (boltz2_reward.py)

Designed to run FROM INSIDE the boltz conda env, e.g. on Colab:
    conda run --no-capture-output -n boltz2 python run_search.py \
        --facts-prefix /content/drive/MyDrive/.../universe_facts/facts \
        --work-dir     /content/drive/MyDrive/.../boltz_search_run \
        --log          /content/drive/MyDrive/.../boltz_search_run/scores.csv

Everything the search touches (work_dir, log, results) should live on Drive so
a Colab disconnect doesn't lose the expensive Boltz-2 outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle

import numpy as np

import boltz2_reward as reward
import build_universe_bitmap as bub
from pool_search import SearchConfig, run_search, BitmapSafetyChecker

# --- fixed problem definition (must match how the facts were built) ---------
ROOT = reward.ROOT_CDRH3                    # "ARERGIHYYGSSEIQDY"
DESIGN_POS = (3, 4, 5, 6, 12, 13, 14)       # 0-based indices into ROOT
N_DESIGN = len(DESIGN_POS)
UNIVERSE_TOTAL = 20 ** N_DESIGN             # 20^7 = 1,280,000,000

# --- safety policy (VERIFIED against the facts: no Ngly/xCys + high/med/low<=3
#     + hydrophobic patch < 5 keeps 817,646,288 of 20^7, 63.88%) --------------
N_HIGH = 3
N_MEDIUM = 3
N_LOW = 3
MAX_ALLOWED_PATCH_LEN = 5   # policy is: max_patch_len < this (i.e. < 5)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Boltz-2 pool search.")
    p.add_argument("--facts-prefix", required=True,
                   help="Path prefix for the precomputed universe facts "
                        "(the part before '_has_ngly.bin', e.g. '.../universe_facts/facts').")
    p.add_argument("--work-dir", required=True,
                   help="Directory for per-generation Boltz YAML inputs + outputs (put on Drive).")
    p.add_argument("--log", default=None,
                   help="Path for the writable per-candidate score CSV (put on Drive).")
    p.add_argument("--results", default=None,
                   help="Path for the final sorted results CSV. Defaults to <work-dir>/results.csv")
    p.add_argument("--max-calls", type=int, default=3000, help="Boltz-2 call budget.")
    p.add_argument("--patience", type=int, default=8, help="Generations without improvement before stopping.")
    p.add_argument("--k-parents", type=int, default=8)
    p.add_argument("--k-children", type=int, default=8)
    p.add_argument("--fitness-gate-tau", type=float, default=None,
                   help="Optional quality gate (MADs below median). Off by default.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility.")
    # Boltz knobs (forwarded to boltz predict); defaults match the validated notebook.
    p.add_argument("--diffusion-samples", type=int, default=1)
    p.add_argument("--diffusion-samples-affinity", type=int, default=5)
    p.add_argument("--max-parallel-samples", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--preprocessing-threads", type=int, default=8)
    p.add_argument("--no-msa-server", action="store_true",
                   help="Disable --use_msa_server (only if you supply MSAs another way).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Sanity: facts must describe THIS space, or every safety lookup is garbage.
    ngly_path = f"{args.facts_prefix}_has_ngly.bin"
    if not os.path.exists(ngly_path):
        raise FileNotFoundError(f"No facts found at prefix '{args.facts_prefix}' (missing {ngly_path}).")
    ngly_bytes = os.path.getsize(ngly_path)
    if ngly_bytes != UNIVERSE_TOTAL:
        raise ValueError(
            f"Facts size {ngly_bytes:,} != expected 20^{N_DESIGN} = {UNIVERSE_TOTAL:,}. "
            f"The facts were built for a different design space than DESIGN_POS={DESIGN_POS}."
        )
    if len(ROOT) <= max(DESIGN_POS):
        raise ValueError("DESIGN_POS indexes past the end of ROOT.")

    print(f"Root: {ROOT} | designable positions: {DESIGN_POS}")
    print(f"Policy: no Ngly/xCys, high<={N_HIGH}, medium<={N_MEDIUM}, low<={N_LOW}, "
          f"patch<{MAX_ALLOWED_PATCH_LEN}")

    facts = bub.load_facts_memmap(args.facts_prefix, UNIVERSE_TOTAL)
    safety = BitmapSafetyChecker(
        facts, n_high=N_HIGH, n_medium=N_MEDIUM, n_low=N_LOW,
        max_allowed_patch_len=MAX_ALLOWED_PATCH_LEN,
    )

    os.makedirs(args.work_dir, exist_ok=True)
    score_fn = reward.Boltz2ScoreFn(
        work_dir=args.work_dir,
        diffusion_samples=args.diffusion_samples,
        diffusion_samples_affinity=args.diffusion_samples_affinity,
        max_parallel_samples=args.max_parallel_samples,
        num_workers=args.num_workers,
        preprocessing_threads=args.preprocessing_threads,
        use_msa_server=not args.no_msa_server,
    )

    config = SearchConfig(
        root_sequence=ROOT,
        designable_positions=DESIGN_POS,
        score_sign=1,                       # higher S(x) = better (verified vs the reward wrapper)
        k_parents=args.k_parents,
        k_children_per_parent=args.k_children,
        patience_generations=args.patience,
        max_boltz2_calls=args.max_calls,
        fitness_gate_tau=args.fitness_gate_tau,
    )

    log_path = args.log
    results_path = args.results or os.path.join(args.work_dir, "results.csv")

    result = run_search(config, score_fn, safety,
                        rng=np.random.default_rng(args.seed), log_path=log_path)

    pool = result["pool"]
    print(f"\nDone. Boltz-2 calls: {result['calls_made']} | pool size: {len(pool)}")

    # Durable outputs go next to --results (point that at Drive); work_dir can be
    # fast local scratch. Full pickle + per-generation log + sorted results CSV.
    results_dir = os.path.dirname(os.path.abspath(results_path))
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "result.pkl"), "wb") as f:
        pickle.dump(result, f)
    with open(os.path.join(results_dir, "log.json"), "w") as f:
        json.dump(result["log"], f, indent=2)

    ranked = sorted(pool, key=lambda m: config.score_sign * m.score, reverse=True)
    with open(results_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "design", "S_score", "boltz_affinity_value", "boltz_probability",
                    "generation", "parent_index", "step", "move_type"])
        for i, m in enumerate(ranked):
            w.writerow([i, m.design, m.score, m.affinity_value, m.affinity_probability,
                        m.generation, m.parent_index, m.step, m.move_type])

    best = ranked[0]
    print(f"Best: design={best.design}  S={best.score:.4f}  "
          f"affinity_value={best.affinity_value}  probability={best.affinity_probability}")
    print(f"Results CSV: {results_path}")
    print(f"Per-candidate log: {log_path}")


if __name__ == "__main__":
    main()
