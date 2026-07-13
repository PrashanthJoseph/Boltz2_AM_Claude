from __future__ import annotations

import re
import numpy as np
import pandas as pd
from typing import Optional, Sequence, Set, Dict, Tuple

# =========================================================
# CDR hydrophobic counts
# =========================================================

DEFAULT_HYDROPHOBIC: Set[str] = set("AVILMFWY")

def add_cdr_hydrophobic_counts_df(
    df: pd.DataFrame,
    *,
    light_col: str = "light_chain",
    heavy_col: str = "heavy_chain",
    hydrophobic: Set[str] = DEFAULT_HYDROPHOBIC,
    prefix: str = "hydro_",
    add_cdr_seqs: bool = False,
    cdr_seq_prefix: str = "cdrseq_",
    on_error: str = "nan",  # "nan" | "zero" | "raise"
) -> pd.DataFrame:
    if on_error not in {"nan", "zero", "raise"}:
        raise ValueError("on_error must be one of: 'nan', 'zero', 'raise'")

    extractor = IMGTRegionExtractor()

    def clean_seq(s: str) -> str:
        s = (s or "").strip().upper()
        return re.sub(r"[^A-Z]", "", s)

    def count_hydro(subseq: str) -> int:
        return sum(1 for aa in subseq if aa in hydrophobic)

    def fail_val():
        if on_error == "zero":
            return 0
        return pd.NA

    light_cdrs = ["cdrl1", "cdrl2", "cdrl3"]
    heavy_cdrs = ["cdrh1", "cdrh2", "cdrh3"]

    for name in light_cdrs + heavy_cdrs:
        col = f"{prefix}{name}"
        if col not in df.columns:
            df[col] = pd.NA
        if add_cdr_seqs:
            scol = f"{cdr_seq_prefix}{name}"
            if scol not in df.columns:
                df[scol] = pd.NA

    for col in [
        "total_hydrophobic_in_cdrls",
        "total_hydrophobic_in_cdrhs",
        "total_hydrophobic_in_all_cdrs",
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    def process_chain(seq: str, chain: str) -> Tuple[Dict[str, int], Dict[str, str]]:
        seq = clean_seq(seq)
        regions = extractor.extract(seq, chain_label=chain).regions

        counts = {n: 0 for n in (light_cdrs if chain == "L" else heavy_cdrs)}
        seqs = {n: "" for n in (light_cdrs if chain == "L" else heavy_cdrs)}

        for r in regions:
            if r.name in counts:
                s0, e0 = r.slice0()
                cdr = seq[s0:e0]
                seqs[r.name] = cdr
                counts[r.name] = count_hydro(cdr)

        return counts, seqs

    for idx, row in df.iterrows():
        l_seq_raw = str(row.get(light_col, "") or "")
        h_seq_raw = str(row.get(heavy_col, "") or "")

        try:
            if l_seq_raw.strip():
                l_counts, l_seqs = process_chain(l_seq_raw, chain="L")
            else:
                l_counts = {n: 0 for n in light_cdrs}
                l_seqs = {n: "" for n in light_cdrs}

            for n in light_cdrs:
                df.at[idx, f"{prefix}{n}"] = l_counts[n]
                if add_cdr_seqs:
                    df.at[idx, f"{cdr_seq_prefix}{n}"] = l_seqs[n]

            l_total = sum(l_counts.values())
            df.at[idx, "total_hydrophobic_in_cdrls"] = l_total
        except Exception as e:
            print(e)
            if on_error == "raise":
                raise
            v = fail_val()
            for n in light_cdrs:
                df.at[idx, f"{prefix}{n}"] = v
                if add_cdr_seqs:
                    df.at[idx, f"{cdr_seq_prefix}{n}"] = v
            df.at[idx, "total_hydrophobic_in_cdrls"] = v
            l_total = v

        try:
            if h_seq_raw.strip():
                h_counts, h_seqs = process_chain(h_seq_raw, chain="H")
            else:
                h_counts = {n: 0 for n in heavy_cdrs}
                h_seqs = {n: "" for n in heavy_cdrs}

            for n in heavy_cdrs:
                df.at[idx, f"{prefix}{n}"] = h_counts[n]
                if add_cdr_seqs:
                    df.at[idx, f"{cdr_seq_prefix}{n}"] = h_seqs[n]

            h_total = sum(h_counts.values())
            df.at[idx, "total_hydrophobic_in_cdrhs"] = h_total
        except Exception:
            if on_error == "raise":
                raise
            v = fail_val()
            for n in heavy_cdrs:
                df.at[idx, f"{prefix}{n}"] = v
                if add_cdr_seqs:
                    df.at[idx, f"{cdr_seq_prefix}{n}"] = v
            df.at[idx, "total_hydrophobic_in_cdrhs"] = v
            h_total = v

        if pd.isna(l_total) or pd.isna(h_total):
            df.at[idx, "total_hydrophobic_in_all_cdrs"] = pd.NA
        else:
            df.at[idx, "total_hydrophobic_in_all_cdrs"] = int(l_total) + int(h_total)

    return df


# =========================================================
# Whole-chain hydrophobic patches
# =========================================================

WINDOW = 5
THRESHOLD = 1.5
MIN_PATCH_LEN = 5

KD_SCALE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": 2.3, "Y": -1.3
}

def clean_seq(seq):
    if pd.isna(seq):
        return ""
    return str(seq).replace(" ", "").replace("\n", "").strip().upper()

def sliding_window_scores(seq):
    scores = []
    for i in range(len(seq) - WINDOW + 1):
        window = seq[i:i+WINDOW]
        val = sum(KD_SCALE.get(a, 0) for a in window) / WINDOW
        scores.append((i + 1, i + WINDOW, val))
    return scores

def find_patches(seq):
    seq = clean_seq(seq)
    if len(seq) < WINDOW:
        return []

    windows = sliding_window_scores(seq)
    flagged = [(s, e, v) for (s, e, v) in windows if v >= THRESHOLD]

    if not flagged:
        return []

    patches = []
    cur_s, cur_e = flagged[0][0], flagged[0][1]

    for s, e, _ in flagged[1:]:
        if s <= cur_e + 1:
            cur_e = max(cur_e, e)
        else:
            if (cur_e - cur_s + 1) >= MIN_PATCH_LEN:
                patches.append((cur_s, cur_e))
            cur_s, cur_e = s, e

    if (cur_e - cur_s + 1) >= MIN_PATCH_LEN:
        patches.append((cur_s, cur_e))

    return patches

def patch_summary(seq):
    patches = find_patches(seq)

    if not patches:
        return pd.Series([0, "", None])

    locations = ";".join(f"{s}-{e}" for s, e in patches)
    lengths = [e - s + 1 for s, e in patches]

    return pd.Series([len(patches), locations, max(lengths)])

def add_hydrophobic_patch_columns(
    df: pd.DataFrame,
    *,
    light_col: str = "light_chain",
    heavy_col: str = "heavy_chain",
) -> pd.DataFrame:
    df[light_col] = df[light_col].apply(clean_seq)
    df[heavy_col] = df[heavy_col].apply(clean_seq)

    df[["lc_patch_count", "lc_patch_locations", "lc_max_patch_len"]] = \
        df[light_col].apply(patch_summary)

    df[["hc_patch_count", "hc_patch_locations", "hc_max_patch_len"]] = \
        df[heavy_col].apply(patch_summary)

    return df


# =========================================================
# Vectorized batch scanning (for large candidate sets, e.g. millions of
# designed CDRH3 variants) — pure array ops, no abnumber/IMGT calls.
# =========================================================

def strings_to_char_array(sequences: Sequence[str], length: int) -> np.ndarray:
    """(N,) equal-length strings -> (N, length) array of single characters, fully vectorized."""
    arr = np.asarray(sequences, dtype=f"<U{length}")
    return arr.view("U1").reshape(-1, length)


def _kd_lookup_by_codepoint() -> np.ndarray:
    table = np.zeros(128, dtype=np.float64)
    for aa, val in KD_SCALE.items():
        table[ord(aa)] = val
    return table


_KD_TABLE = _kd_lookup_by_codepoint()


def char_array_to_kd_batch(arr: np.ndarray) -> np.ndarray:
    # 'U1' elements are 4-byte UCS4 codepoints internally, so this view is a
    # zero-copy reinterpretation, not a cast — safe for the plain-ASCII AA alphabet.
    codepoints = arr.view(np.uint32)
    return _KD_TABLE[codepoints]


def windowed_kd_scores_batch(kd_values: np.ndarray, window: int = WINDOW) -> np.ndarray:
    # Neumaier (compensated) summation, not a plain running sum: Python 3.12's
    # built-in sum() — what the original per-window sum(...) uses — applies
    # Neumaier summation for float inputs, which is measurably more accurate
    # than naive sequential float addition. Right at THRESHOLD that accuracy
    # gap can flip a window's flagged status, so a naive vectorized sum
    # silently disagrees with the original on borderline windows. This
    # reproduces the original's rounding bit-for-bit (verified against
    # math.fsum / Python's sum() on adversarial test cases).
    n, length = kd_values.shape
    num_windows = length - window + 1
    total = np.zeros((n, num_windows), dtype=np.float64)
    comp = np.zeros((n, num_windows), dtype=np.float64)
    for offset in range(window):
        x = kd_values[:, offset:offset + num_windows]
        t = total + x
        use_total = np.abs(total) >= np.abs(x)
        comp += np.where(use_total, (total - t) + x, (x - t) + total)
        total = t
    return (total + comp) / window


def max_patch_length_batch(flagged: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """
    Vectorized, exact port of `find_patches`'s merge rule (merge the next
    flagged window into the running patch if its start is within `window` of
    the patch's current end) applied to every row at once. Returns the
    longest qualifying patch length in residues per row, 0 if none flagged.
    """
    n, w = flagged.shape
    NEG = -(10 ** 9)
    last_pos = np.full(n, NEG, dtype=np.int64)
    cluster_start = np.zeros(n, dtype=np.int64)
    best = np.zeros(n, dtype=np.int64)
    for i in range(w):
        is_flagged = flagged[:, i]
        gap_ok = (i - last_pos) <= window
        continues = is_flagged & gap_ok & (last_pos > NEG)
        starts_new = is_flagged & ~continues
        cluster_start = np.where(starts_new, i, cluster_start)
        current_span = np.where(is_flagged, i - cluster_start, 0)
        best = np.maximum(best, current_span)
        last_pos = np.where(is_flagged, i, last_pos)
    any_flag = flagged.any(axis=1)
    return np.where(best > 0, best + window, np.where(any_flag, window, 0))


def hydrophobic_count_batch(arr: np.ndarray, hydrophobic: Set[str] = DEFAULT_HYDROPHOBIC) -> np.ndarray:
    return np.isin(arr, np.array(list(hydrophobic))).sum(axis=1)


def analyze_hydrophobicity_batch(
    sequences,
    *,
    cdr_only_input: bool = True,
    light_col: str = "light_chain",
    heavy_col: str = "heavy_chain",
    hydrophobic: Set[str] = DEFAULT_HYDROPHOBIC,
    window: int = WINDOW,
    threshold: float = THRESHOLD,
    max_allowed_patch_len: int = MIN_PATCH_LEN,
) -> Dict[str, np.ndarray]:
    """
    cdr_only_input=True (default): `sequences` are already the CDR fragment
    (e.g. candidate CDRH3 designs) — pure vectorized array ops, no
    abnumber/IMGT extraction.

    cdr_only_input=False: `sequences` must be a DataFrame with `light_col`/
    `heavy_col` columns; falls back to the original whole-chain
    add_cdr_hydrophobic_counts_df / add_hydrophobic_patch_columns path,
    unchanged, just returned as arrays here.

    max_allowed_patch_len is a REJECTION ceiling ("longest tolerable
    hydrophobic patch"), separate from the module-level MIN_PATCH_LEN, which
    is a DETECTION floor used inside find_patches()/max_patch_length_batch
    ("how long a run of flagged windows must be to count as a patch at all").
    They happen to share a default value (5) but answer different questions
    -- lower max_allowed_patch_len for stricter rejection of shorter patches.
    """
    if not cdr_only_input:
        df = sequences
        df = add_cdr_hydrophobic_counts_df(df, light_col=light_col, heavy_col=heavy_col, hydrophobic=hydrophobic)
        df = add_hydrophobic_patch_columns(df, light_col=light_col, heavy_col=heavy_col)
        return {
            "hydro_total": df["total_hydrophobic_in_all_cdrs"].to_numpy(),
            "lc_max_patch_len": df["lc_max_patch_len"].to_numpy(),
            "hc_max_patch_len": df["hc_max_patch_len"].to_numpy(),
        }

    lengths = {len(s) for s in sequences}
    if len(lengths) != 1:
        raise ValueError(f"cdr_only_input=True requires equal-length sequences, got lengths {lengths}")
    length = lengths.pop()

    arr = strings_to_char_array(sequences, length)
    hydro_total = hydrophobic_count_batch(arr, hydrophobic)

    if length < window:
        max_patch_len = np.zeros(arr.shape[0], dtype=np.int64)
    else:
        kd_values = char_array_to_kd_batch(arr)
        scores = windowed_kd_scores_batch(kd_values, window)
        flagged = scores >= threshold
        max_patch_len = max_patch_length_batch(flagged, window)

    return {
        "hydro_total": hydro_total,
        "max_patch_len": max_patch_len,
        "passes_patch_filter": max_patch_len < max_allowed_patch_len,
    }


# =========================================================
# Wrapper
# =========================================================

def run_hydrophobic_analysis(
    input_csv: str,
    output_csv: str,
    *,
    add_cdr_seqs: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    df = add_cdr_hydrophobic_counts_df(
        df,
        light_col="light_chain",
        heavy_col="heavy_chain",
        add_cdr_seqs=add_cdr_seqs,
        on_error="nan",
    )

    df = add_hydrophobic_patch_columns(
        df,
        light_col="light_chain",
        heavy_col="heavy_chain",
    )

    df.to_csv(output_csv, index=False)
    return df


# =========================================================
# Run
# =========================================================
# if __name__ == "__main__":
#     input_csv = r"/mnt/d/INITO/RFAB_PIPELINE/rfab_antibmpnn_nr_LAP.csv"
#     output_csv = r"/mnt/d/INITO/RFAB_PIPELINE/rfab_antibmpnn_nr_LAP_hydro_patches.csv"
#
#     df = run_hydrophobic_analysis(input_csv, output_csv, add_cdr_seqs=True)
#     print(df.head())
#     print("Done. Saved to:", output_csv)
