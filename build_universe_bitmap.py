from __future__ import annotations

from typing import Sequence

import numpy as np

import LAP_inhouse as lap
import hydrophobic_analysis as hydro

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def build_universe_facts(
    root_sequence: str,
    designable_positions: Sequence[int],
    chunk_size: int = 20_000_000,
    alphabet: str = AA_ALPHABET,
    progress: bool = True,
) -> dict:
    """
    One-time, POLICY-INDEPENDENT scan of the full alphabet^len(designable_positions)
    space. Stores per-sequence facts needed to support ANY severity-count
    policy afterwards without re-scanning: whether Ngly/xCys fired at all
    (hard-reject tier), counts for the remaining High-severity rules /
    Medium / Low (each compared against a threshold later), and longest
    hydrophobic patch length. See LAP_inhouse.passes_lap_policy for the
    policy this is designed to support, and derive_bitmap() for the cheap
    per-policy derivation.
    """
    n_design = len(designable_positions)
    base = len(alphabet)
    total = base ** n_design
    aa_arr = np.array(list(alphabet))
    root_arr = np.array(list(root_sequence))

    has_ngly = np.zeros(total, dtype=bool)
    has_xcys = np.zeros(total, dtype=bool)
    high_count_excl = np.zeros(total, dtype=np.uint8)
    medium_count = np.zeros(total, dtype=np.uint8)
    low_count = np.zeros(total, dtype=np.uint8)
    max_patch_len = np.zeros(total, dtype=np.uint8)

    n_chunks = (total + chunk_size - 1) // chunk_size
    for ci, start in enumerate(range(0, total, chunk_size)):
        end = min(start + chunk_size, total)
        n = end - start

        digits = np.zeros((n, n_design), dtype=np.int64)
        rem = np.arange(start, end, dtype=np.int64)
        for pos in range(n_design):
            digits[:, pos] = rem % base
            rem //= base
        design_chars = aa_arr[digits]

        full_arr = np.tile(root_arr, (n, 1))
        for i, pos in enumerate(designable_positions):
            full_arr[:, pos] = design_chars[:, i]
        full_seqs = ["".join(row) for row in full_arr]

        lap_res = lap.scan_lap_motifs_batch(full_seqs, cdr_only_input=True)
        has_ngly[start:end] = lap_res["per_rule_hits"]["Ngly"]
        has_xcys[start:end] = lap_res["per_rule_hits"]["xCys"]
        high_count_excl[start:end] = lap.high_count_excluding_ngly_xcys(lap_res)
        medium_count[start:end] = lap_res["medium_count"]
        low_count[start:end] = lap_res["low_count"]

        hres = hydro.analyze_hydrophobicity_batch(full_seqs, cdr_only_input=True)
        max_patch_len[start:end] = np.clip(hres["max_patch_len"], 0, 255)

        if progress:
            print(f"chunk {ci + 1}/{n_chunks} done ({end:,}/{total:,})", flush=True)

    return {
        "has_ngly": has_ngly,
        "has_xcys": has_xcys,
        "high_count_excl": high_count_excl,
        "medium_count": medium_count,
        "low_count": low_count,
        "max_patch_len": max_patch_len,
    }


def build_universe_facts_to_disk(
    root_sequence: str,
    designable_positions: Sequence[int],
    path_prefix: str,
    chunk_size: int = 20_000_000,
    alphabet: str = AA_ALPHABET,
    progress: bool = True,
) -> int:
    """
    Same scan as build_universe_facts, but writes each chunk's results
    DIRECTLY to disk-backed memmap files (mode='w+') instead of accumulating
    the full ~7.7GB of output arrays in RAM first. Peak RAM usage is just the
    per-chunk temporary arrays (~chunk_size * a few bytes, tens-to-couple-
    hundred MB), not the full universe -- build_universe_facts alone would
    hold the whole thing in RAM before you ever got to save it, which
    defeats the point if RAM is tight. Returns `total` (needed later to
    reopen the files via load_facts_memmap).
    """
    n_design = len(designable_positions)
    base = len(alphabet)
    total = base ** n_design
    aa_arr = np.array(list(alphabet))
    root_arr = np.array(list(root_sequence))

    out = {name: np.memmap(f"{path_prefix}_{name}.bin", dtype=np.uint8, mode="w+", shape=(total,))
           for name in FACT_NAMES}

    n_chunks = (total + chunk_size - 1) // chunk_size
    for ci, start in enumerate(range(0, total, chunk_size)):
        end = min(start + chunk_size, total)
        n = end - start

        digits = np.zeros((n, n_design), dtype=np.int64)
        rem = np.arange(start, end, dtype=np.int64)
        for pos in range(n_design):
            digits[:, pos] = rem % base
            rem //= base
        design_chars = aa_arr[digits]

        full_arr = np.tile(root_arr, (n, 1))
        for i, pos in enumerate(designable_positions):
            full_arr[:, pos] = design_chars[:, i]
        full_seqs = ["".join(row) for row in full_arr]

        lap_res = lap.scan_lap_motifs_batch(full_seqs, cdr_only_input=True)
        out["has_ngly"][start:end] = lap_res["per_rule_hits"]["Ngly"].astype(np.uint8)
        out["has_xcys"][start:end] = lap_res["per_rule_hits"]["xCys"].astype(np.uint8)
        out["high_count_excl"][start:end] = lap.high_count_excluding_ngly_xcys(lap_res)
        out["medium_count"][start:end] = lap_res["medium_count"]
        out["low_count"][start:end] = lap_res["low_count"]

        hres = hydro.analyze_hydrophobicity_batch(full_seqs, cdr_only_input=True)
        out["max_patch_len"][start:end] = np.clip(hres["max_patch_len"], 0, 255)

        if progress:
            print(f"chunk {ci + 1}/{n_chunks} done ({end:,}/{total:,})", flush=True)

    for name in FACT_NAMES:
        out[name].flush()

    return total


def apply_policy(
    has_ngly: np.ndarray,
    has_xcys: np.ndarray,
    high_count_excl: np.ndarray,
    medium_count: np.ndarray,
    low_count: np.ndarray,
    max_patch_len: np.ndarray,
    *,
    n_high: int,
    n_medium: int,
    n_low: int,
    max_allowed_patch_len: int,
) -> np.ndarray:
    """
    Core policy logic, factored out so it works identically whether given
    the FULL universe's fact arrays or a small already-looked-up subset (see
    BitmapSafetyChecker, which only ever passes small subsets so it never
    has to materialize the full arrays). has_ngly/has_xcys are 0/1 uint8
    here (not bool), so this uses `== 0` rather than `~x` -- correct either
    way, but explicit equality avoids relying on bitwise-NOT-on-uint8 quirks.
    Mirrors LAP_inhouse.passes_lap_policy's two-stage logic exactly (hard
    reject on Ngly/xCys, then threshold the remaining counts).
    """
    keep = (has_ngly == 0) & (has_xcys == 0)
    keep &= high_count_excl <= n_high
    keep &= medium_count <= n_medium
    keep &= low_count <= n_low
    keep &= max_patch_len < max_allowed_patch_len
    return keep


def derive_bitmap(
    facts: dict,
    *,
    n_high: int,
    n_medium: int,
    n_low: int,
    max_allowed_patch_len: int = hydro.MIN_PATCH_LEN,
) -> np.ndarray:
    """Applies a policy over the FULL facts arrays at once -- fine for
    small/test-scale facts kept in RAM, but touches every element, so NOT
    what BitmapSafetyChecker uses for the full 20^7 universe (that does
    sparse per-query lookups instead, see load_facts_memmap)."""
    return apply_policy(
        facts["has_ngly"], facts["has_xcys"], facts["high_count_excl"],
        facts["medium_count"], facts["low_count"], facts["max_patch_len"],
        n_high=n_high, n_medium=n_medium, n_low=n_low, max_allowed_patch_len=max_allowed_patch_len,
    )


def encode_design_batch(design_chars: np.ndarray, alphabet: str = AA_ALPHABET) -> np.ndarray:
    """(N, L) char array -> (N,) int64 base-len(alphabet) encoded indices, vectorized."""
    char_to_idx = {c: i for i, c in enumerate(alphabet)}
    base = len(alphabet)
    n, length = design_chars.shape
    idx = np.zeros(n, dtype=np.int64)
    lookup = np.zeros(128, dtype=np.int64)
    for c, i in char_to_idx.items():
        lookup[ord(c)] = i
    codepoints = design_chars.view(np.uint32)
    digit_values = lookup[codepoints]
    for pos in range(length):
        idx += digit_values[:, pos] * (base ** pos)
    return idx


FACT_NAMES = ("has_ngly", "has_xcys", "high_count_excl", "medium_count", "low_count", "max_patch_len")


def save_facts(facts: dict, path_prefix: str) -> None:
    # Deliberately NOT bit-packed: one byte per entry for every fact,
    # including the two boolean ones. Bit-packing has_ngly/has_xcys would
    # save disk space (~1.1GB), but would require unpacking bit-offset math
    # on every sparse lookup, which defeats memmap's whole purpose (only
    # touch the pages you actually query). Disk is cheap; simplicity of the
    # sparse-lookup path is worth more here.
    for name in FACT_NAMES:
        arr = facts[name]
        if arr.dtype == bool:
            arr = arr.astype(np.uint8)
        arr.tofile(f"{path_prefix}_{name}.bin")


def load_facts(path_prefix: str, total: int) -> dict:
    """Fully loads all facts into RAM. Fine for small/test-scale facts;
    for the full 20^7 universe use load_facts_memmap instead."""
    return {name: np.fromfile(f"{path_prefix}_{name}.bin", dtype=np.uint8) for name in FACT_NAMES}


def load_facts_memmap(path_prefix: str, total: int) -> dict:
    """Memory-mapped load: none of the ~5-7GB of fact arrays are read into
    RAM until specific indices are actually queried (via BitmapSafetyChecker,
    which only ever indexes the handful of candidates proposed each
    generation) -- correct even when the files are far larger than available
    RAM."""
    return {
        name: np.memmap(f"{path_prefix}_{name}.bin", dtype=np.uint8, mode="r", shape=(total,))
        for name in FACT_NAMES
    }
