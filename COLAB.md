# Running the Boltz-2 pool search on Colab (A100)

Clone-and-run guide. Paste each block into its own Colab cell, in order.
Use an **A100** runtime (Runtime → Change runtime type → A100). 8 GB cards
cannot fit the ~230-residue + ligand affinity job.

The search runs *inside* the `boltz2` conda env and shells out to `boltz predict`
per generation. Safety is checked against the **precomputed universe facts** (no
`abnumber`/LAP at runtime). The facts were built for `root="ARERGIHYYGSSEIQDY"`,
`designable_positions=(3,4,5,6,12,13,14)` — the search must use the same, which
`run_search.py` hardcodes and sanity-checks against the facts' file size.

---

## Cell 1 — clone the repo

```python
!git clone https://github.com/PrashanthJoseph/Boltz2_AM_Claude.git /content/Boltz2_AM_Claude
```

## Cell 2 — Miniconda + isolated env

```python
!wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
!bash ./miniconda.sh -b -f -p /usr/local
import os
os.environ['PATH'] = '/usr/local/bin:' + os.environ['PATH']
!conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
!conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
!conda create -n boltz2 python=3.10 -y
```

## Cell 3 — install boltz + GPU kernels + the search's deps (one env)

```bash
%%bash
set -e
conda run -n boltz2 pip install -q boltz -U
conda run -n boltz2 pip install -q "triton==3.3.0" cuequivariance-torch cuequivariance-ops-torch-cu12
# search side (numpy comes with boltz; pyyaml for YAML writing). NO abnumber needed.
conda run -n boltz2 pip install -q pyyaml numpy
```

## Cell 4 — sanity check (confirm A100 + kernel import)

```bash
%%bash
conda run -n boltz2 python - << 'PY'
import torch
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
from cuequivariance_torch import triangle_multiplicative_update
print("triangle_multiplicative_update OK")
PY
```

## Cell 5 — mount Drive and stage the facts locally

Upload the 6 `universe_facts/facts_*.bin` files (~7.2 GB total) to Drive once,
then copy them to Colab's **local** disk — random-access memmap over the Drive
FUSE mount is slow/unreliable, local is fast.

```python
from google.colab import drive
drive.mount('/content/drive')

import shutil, os
# EDIT this to wherever you uploaded universe_facts/ on Drive:
DRIVE_FACTS = "/content/drive/MyDrive/Project_Alphafold/BOLTZ/universe_facts"
os.makedirs("/content/universe_facts", exist_ok=True)
for name in ("has_ngly","has_xcys","high_count_excl","medium_count","low_count","max_patch_len"):
    src = f"{DRIVE_FACTS}/facts_{name}.bin"
    dst = f"/content/universe_facts/facts_{name}.bin"
    if not os.path.exists(dst):
        print("copying", name, "..."); shutil.copy(src, dst)
print("facts staged at /content/universe_facts")
```

## Cell 6 — run the search

`--work-dir` is fast local Boltz scratch; `--log` and `--results` go to Drive so
a disconnect keeps the per-candidate scores and final ranking.

```bash
%%bash
DRIVE_OUT=/content/drive/MyDrive/Project_Alphafold/BOLTZ/boltz_search_run
mkdir -p "$DRIVE_OUT"
conda run --no-capture-output -n boltz2 python /content/Boltz2_AM_Claude/run_search.py \
  --facts-prefix /content/universe_facts/facts \
  --work-dir     /content/boltz_scratch \
  --log          "$DRIVE_OUT/scores.csv" \
  --results      "$DRIVE_OUT/results.csv" \
  --max-calls 3000 --patience 8 \
  --k-parents 8 --k-children 8
```

Outputs (all on Drive next to `results.csv`): `scores.csv` (per-candidate:
Boltz affinity value + probability + S(x) + normalized_objective, written live),
`results.csv` (final ranking), `log.json` (per-generation log), `result.pkl`.

---

## Notes / gotchas

- **Verify the affinity output path on the first batch.** `read_affinity_json`
  tries `<out>/predictions/<id>/affinity_<id>.json` then falls back to a
  recursive glob. If the very first generation errors with "No affinity output",
  Boltz's layout differs — check `boltz_scratch/batch_000000/out/` and tell me
  the actual path.
- **MSA server rate limits.** `--use_msa_server` is on. Across ~3000 candidates
  the public server may throttle. If the run stalls on MSA, that's why — we'd
  switch to a reused/precomputed MSA (the framework is fixed; only CDRH3 varies).
- **Time.** ~230-residue complex + affinity (5 affinity samples) on A100 is
  roughly 1–2 min/candidate → order of days for 3000 calls. `--patience 8` will
  usually stop earlier. Start with a small `--max-calls` (e.g. 40) to calibrate.
- **Cache the env to Drive** to skip the ~10 min rebuild on reconnect: after
  Cell 3, `tar czf` `/usr/local/envs/boltz2` to Drive, and restore + `tar xzf`
  on the next session instead of re-running Cells 2–3.
- **`--fitness-gate-tau`** (default off) enables the quality gate (drop parents
  more than τ MADs below the pool median); pass e.g. `--fitness-gate-tau 2.5`.
```
