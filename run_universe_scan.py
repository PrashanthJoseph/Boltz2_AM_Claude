import sys
import time
import types

sys.path.insert(0, r"D:\INITO\myProjects\Estradiol\e2pico_antibodies\Boltz2_AM_Claude")

try:
    import abnumber  # noqa: F401
except ImportError:
    # LAP_inhouse.py imports abnumber.Chain at module level, but the
    # cdr_only_input=True path used here never actually calls it -- stub it
    # out so this script doesn't need abnumber installed just to run the
    # vectorized, abnumber-free scan.
    stub = types.ModuleType("abnumber")

    class _UnusedChain:
        def __init__(self, *a, **kw):
            raise NotImplementedError("abnumber stub -- not used by cdr_only_input=True path")

    stub.Chain = _UnusedChain
    sys.modules["abnumber"] = stub

import build_universe_bitmap as bub

ROOT = "ARERGIHYYGSSEIQDY"
DESIGN_POS = (3, 4, 5, 6, 12, 13, 14)
OUT_PREFIX = r"D:\INITO\myProjects\Estradiol\e2pico_antibodies\Boltz2_AM_Claude\universe_facts\facts"

if __name__ == "__main__":
    t0 = time.time()
    print(f"Starting full-universe scan: root={ROOT}, designable_positions={DESIGN_POS}", flush=True)
    total = bub.build_universe_facts_to_disk(
        ROOT, DESIGN_POS, OUT_PREFIX, chunk_size=20_000_000, progress=True
    )
    elapsed = time.time() - t0
    print(f"DONE. total={total:,} sequences. elapsed={elapsed/3600:.2f} hours", flush=True)
    print(f"Facts saved to: {OUT_PREFIX}_*.bin", flush=True)
