from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal, Any, Sequence, FrozenSet, Iterable

import numpy as np
import pandas as pd
from abnumber import Chain


# ============================================================
# FASTA: records contain "LCSEQ:HCSEQ" (separated by ':')
# ============================================================

def fasta_lc_hc_to_dict(fasta_path: str) -> Dict[str, Tuple[str, str]]:
    """
    Parses a FASTA file where each record's sequence is formatted as:
      <light_chain_sequence>:<heavy_chain_sequence>

    Returns:
      {record_id: (light_seq, heavy_seq)}

    NOTE:
      - The sequences are uppercased and stripped to A-Z and ':' only.
      - If there are multiple ':' characters, this raises ValueError.
    """
    records: Dict[str, str] = {}
    with open(fasta_path, "r", encoding="utf-8") as f:
        cur_id: Optional[str] = None
        cur_seq: List[str] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    records[cur_id] = "".join(cur_seq)
                cur_id = line[1:].strip()
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id is not None:
            records[cur_id] = "".join(cur_seq)

    out: Dict[str, Tuple[str, str]] = {}
    for rid, raw in records.items():
        raw = raw.strip().upper()
        raw = re.sub(r"[^A-Z:]", "", raw)
        parts = raw.split(":")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                f"Record '{rid}': expected LC:HC (single ':'), got: '{raw[:120]}'"
            )
        out[rid] = (parts[0], parts[1])
    return out


# ============================================================
# Region model
# ============================================================

@dataclass(frozen=True)
class Region:
    """
    1-based inclusive coordinates in the original chain sequence.
    """
    name: str
    start: int
    end: int

    def slice0(self) -> Tuple[int, int]:
        return (self.start - 1, self.end)  # 0-based half-open


@dataclass(frozen=True)
class ChainRegions:
    chain: Literal["L", "H"]
    regions: List[Region]  # fwl1/cdrl1/... or fwh1/cdrh1/... in order

    def by_prefix(self, prefix: str) -> List[Region]:
        return [r for r in self.regions if r.name.startswith(prefix)]


# ============================================================
# IMGT-based region extraction (abnumber)
# ============================================================
class IMGTRegionExtractor:
    """
    Uses abnumber Chain(..., scheme="imgt") and its *_seq attributes
    (fr1_seq, cdr1_seq, ..., fr4_seq) to build coordinate Regions.

    Works with abnumber versions that expose ch.cdr1_seq etc.
    """

    def extract(self, seq: str, chain_label: Literal["L", "H"]) -> ChainRegions:
        seq_clean = re.sub(r"[^A-Z]", "", (seq or "").strip().upper())
        ch = Chain(seq_clean, scheme="imgt")

        if chain_label == "L":
            fw_prefix, cdr_prefix = "fwl", "cdrl"
        else:
            fw_prefix, cdr_prefix = "fwh", "cdrh"

        # Pull region AA strings directly from abnumber
        parts = [
            (f"{fw_prefix}1", self._clean_region(getattr(ch, "fr1_seq", ""))),
            (f"{cdr_prefix}1", self._clean_region(getattr(ch, "cdr1_seq", ""))),
            (f"{fw_prefix}2", self._clean_region(getattr(ch, "fr2_seq", ""))),
            (f"{cdr_prefix}2", self._clean_region(getattr(ch, "cdr2_seq", ""))),
            (f"{fw_prefix}3", self._clean_region(getattr(ch, "fr3_seq", ""))),
            (f"{cdr_prefix}3", self._clean_region(getattr(ch, "cdr3_seq", ""))),
            (f"{fw_prefix}4", self._clean_region(getattr(ch, "fr4_seq", ""))),
        ]

        regions: List[Region] = []
        search_start = 0  # 0-based index for ordered find

        for name, region_seq in parts:
            if not region_seq:
                continue

            idx = seq_clean.find(region_seq, search_start)
            if idx == -1:
                # If this happens, it usually means abnumber trimmed/normalized something.
                # Raise with context so you can see what failed.
                raise ValueError(
                    f"{chain_label}: could not locate {name} region sequence in input. "
                    f"Looking for '{region_seq[:20]}...{region_seq[-20:]}' starting at {search_start}."
                )

            start1 = idx + 1
            end1 = idx + len(region_seq)
            regions.append(Region(name=name, start=start1, end=end1))

            search_start = idx + len(region_seq)  # enforce correct ordering

        return ChainRegions(chain=chain_label, regions=regions)

    @staticmethod
    def _clean_region(s: str) -> str:
        # Keep only letters; abnumber strings are usually clean already, but safe.
        return re.sub(r"[^A-Z]", "", (s or "").upper())

# ============================================================
# Motif rules
# ============================================================

Severity = Literal["High", "Medium", "Low"]
RegionScope = Literal["cdr", "all"]  # scan only CDRs or all regions in the chain


@dataclass(frozen=True)
class MotifRule:
    name: str
    flag: str
    severity: Severity
    motif: str
    scope: RegionScope
    pattern: re.Pattern
    # Vectorized-scan equivalents of `pattern`, kept in sync manually — every
    # rule below is a fixed-width character-class motif, so both forms exist.
    position_classes: Optional[Tuple[FrozenSet[str], ...]] = None
    alt_literals: Optional[Tuple[str, ...]] = None


ALL_AA: FrozenSet[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")


def build_rules() -> List[MotifRule]:
    # NOTE: Everything here is regex-based.
    # We also add xCys separately as a high-risk rule computed per CDR residue.
    return [
        MotifRule("Deamidation (high)", "DeAmdH", "High",    "N[GS] (CDRs)",                 "cdr", re.compile(r"N[GS]"),
                  position_classes=(frozenset("N"), frozenset("GS"))),
        MotifRule("Fragmentation (high)", "FragH", "High",   "DP (CDRs)",                    "cdr", re.compile(r"DP"),
                  position_classes=(frozenset("D"), frozenset("P"))),
        MotifRule("Isomerization", "Isom", "High",           "D[DGHST] (CDRs)",              "cdr", re.compile(r"D[DGHST]"),
                  position_classes=(frozenset("D"), frozenset("DGHST"))),
        MotifRule("N-linked glycosylation", "Ngly", "High",  "N[^P][ST] (whole chain)",      "all", re.compile(r"N[^P][ST]"),
                  position_classes=(frozenset("N"), ALL_AA - frozenset("P"), frozenset("ST"))),
        MotifRule("Deamidation (medium)", "DeAmdM", "Medium","N[AHNT] (CDRs)",              "cdr", re.compile(r"N[AHNT]"),
                  position_classes=(frozenset("N"), frozenset("AHNT"))),
        MotifRule("Hydrolysis", "Hydro", "Medium",           "NP (CDRs)",                    "cdr", re.compile(r"NP"),
                  position_classes=(frozenset("N"), frozenset("P"))),
        MotifRule("Fragmentation (medium)", "FragM", "Medium","TS (CDRs)",                  "cdr", re.compile(r"TS"),
                  position_classes=(frozenset("T"), frozenset("S"))),
        MotifRule("Trp oxidation", "TrpOx", "Medium",        "W (CDRs)",                     "cdr", re.compile(r"W"),
                  position_classes=(frozenset("W"),)),
        MotifRule("Met oxidation", "MetOx", "Medium",        "M (CDRs)",                     "cdr", re.compile(r"M"),
                  position_classes=(frozenset("M"),)),
        MotifRule("Deamidation (low)", "DeAmdL", "Low",      "[STK]N (CDRs)",                "cdr", re.compile(r"[STK]N"),
                  position_classes=(frozenset("STK"), frozenset("N"))),
        MotifRule("Integrin binding", "IntBind", "Low",      "GPR|RGD|RYD|LDV|DGE|KGD|NGR",  "all", re.compile(r"(?:GPR|RGD|RYD|LDV|DGE|KGD|NGR)"),
                  alt_literals=("GPR", "RGD", "RYD", "LDV", "DGE", "KGD", "NGR")),
    ]

# ============================================================
# Vectorized batch scanning (for large candidate sets, e.g. millions of
# designed CDRH3 variants) — pure array ops, no abnumber/IMGT calls.
# ============================================================

def strings_to_char_array(sequences: Sequence[str], length: int) -> np.ndarray:
    """(N,) equal-length strings -> (N, length) array of single characters, fully vectorized."""
    arr = np.asarray(sequences, dtype=f"<U{length}")
    return arr.view("U1").reshape(-1, length)


def scan_fixed_motif(arr: np.ndarray, position_classes: Sequence[Iterable[str]]) -> np.ndarray:
    """(N, L) char array -> (N,) bool: True where the fixed-width motif matches at any offset."""
    n, length = arr.shape
    k = len(position_classes)
    if k > length:
        return np.zeros(n, dtype=bool)
    classes = [np.array(list(c)) for c in position_classes]
    hit = np.zeros(n, dtype=bool)
    for start in range(length - k + 1):
        window_ok = np.isin(arr[:, start], classes[0])
        for offset in range(1, k):
            window_ok &= np.isin(arr[:, start + offset], classes[offset])
        hit |= window_ok
    return hit


def scan_literal_alternatives(arr: np.ndarray, literals: Sequence[str]) -> np.ndarray:
    hit = np.zeros(arr.shape[0], dtype=bool)
    for lit in literals:
        hit |= scan_fixed_motif(arr, [frozenset(c) for c in lit])
    return hit


def scan_extra_cysteine_batch(arr: np.ndarray) -> np.ndarray:
    return (arr == "C").any(axis=1)


def scan_lap_motifs_batch(
    sequences: Sequence[str],
    *,
    cdr_only_input: bool = True,
    rules: Optional[List[MotifRule]] = None,
    include_extra_cysteine: bool = True,
    light_seqs: Optional[Sequence[str]] = None,
    heavy_seqs: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Vectorized LAP-motif scan over a batch of sequences.

    cdr_only_input=True (default): `sequences` are already the CDR fragment to
    scan (e.g. candidate CDRH3 designs) — no abnumber/IMGT extraction, pure
    array ops. All rules (including ones tagged scope="all" in build_rules)
    are scanned directly against the given fragment, since there is no
    framework sequence supplied in this mode to extend the scan into.

    cdr_only_input=False: falls back to the original per-row IMGT-based
    AntibodyMotifAnnotator path (`light_seqs`/`heavy_seqs` required) —
    unchanged behavior, just batched into arrays here.

    Returns per-rule hit arrays plus per-severity counts, shape (N,).
    """
    rules = rules or build_rules()

    if not cdr_only_input:
        if light_seqs is None or heavy_seqs is None:
            raise ValueError("cdr_only_input=False requires light_seqs and heavy_seqs")
        annotator = AntibodyMotifAnnotator(rules=rules)
        n = len(light_seqs)
        high = np.zeros(n, dtype=np.int32)
        med = np.zeros(n, dtype=np.int32)
        low = np.zeros(n, dtype=np.int32)
        for i, (lc, hc) in enumerate(zip(light_seqs, heavy_seqs)):
            _, summary = annotator.annotate_one(str(i), lc, hc)
            high[i] = summary["HighRiskCount"]
            med[i] = summary["MediumRiskCount"]
            low[i] = summary["LowRiskCount"]
        return {"high_count": high, "medium_count": med, "low_count": low}

    lengths = {len(s) for s in sequences}
    if len(lengths) != 1:
        raise ValueError(f"cdr_only_input=True requires equal-length sequences, got lengths {lengths}")
    length = lengths.pop()
    arr = strings_to_char_array(sequences, length)

    per_rule_hits: Dict[str, np.ndarray] = {}
    severity_by_flag: Dict[str, Severity] = {}
    for rule in rules:
        if rule.alt_literals is not None:
            per_rule_hits[rule.flag] = scan_literal_alternatives(arr, rule.alt_literals)
        elif rule.position_classes is not None:
            per_rule_hits[rule.flag] = scan_fixed_motif(arr, rule.position_classes)
        else:
            raise ValueError(f"Rule {rule.name!r} has no vectorized form (position_classes/alt_literals)")
        severity_by_flag[rule.flag] = rule.severity

    if include_extra_cysteine:
        per_rule_hits["xCys"] = scan_extra_cysteine_batch(arr)
        severity_by_flag["xCys"] = "High"

    n = arr.shape[0]
    high_count = np.zeros(n, dtype=np.int32)
    med_count = np.zeros(n, dtype=np.int32)
    low_count = np.zeros(n, dtype=np.int32)
    for flag, hits in per_rule_hits.items():
        sev = severity_by_flag[flag]
        if sev == "High":
            high_count += hits
        elif sev == "Medium":
            med_count += hits
        else:
            low_count += hits

    return {
        "per_rule_hits": per_rule_hits,
        "high_count": high_count,
        "medium_count": med_count,
        "low_count": low_count,
    }


def passes_lap_filter(
    sequences: Sequence[str],
    *,
    reject_severities: Tuple[Severity, ...] = ("High",),
    rules: Optional[List[MotifRule]] = None,
    include_extra_cysteine: bool = True,
) -> np.ndarray:
    """(N,) bool — True where the sequence has zero hits in every tier listed in `reject_severities`."""
    result = scan_lap_motifs_batch(
        sequences, cdr_only_input=True, rules=rules, include_extra_cysteine=include_extra_cysteine
    )
    keep = np.ones(len(sequences), dtype=bool)
    if "High" in reject_severities:
        keep &= result["high_count"] == 0
    if "Medium" in reject_severities:
        keep &= result["medium_count"] == 0
    if "Low" in reject_severities:
        keep &= result["low_count"] == 0
    return keep


def high_count_excluding_ngly_xcys(result: Dict[str, Any]) -> np.ndarray:
    """result: output of scan_lap_motifs_batch. Ngly/xCys are handled as a
    separate hard-reject tier (see passes_lap_policy) -- this is the count of
    the OTHER High-severity rules only (currently DeAmdH, FragH, Isom)."""
    has_ngly = result["per_rule_hits"]["Ngly"].astype(np.int32)
    has_xcys = result["per_rule_hits"]["xCys"].astype(np.int32)
    return result["high_count"] - has_ngly - has_xcys


def passes_lap_policy(
    sequences: Sequence[str],
    *,
    n_high: int,
    n_medium: int,
    n_low: int,
    rules: Optional[List[MotifRule]] = None,
) -> np.ndarray:
    """
    (N,) bool. Two-stage policy:
      1. Hard reject: any Ngly (N-linked glycosylation) or xCys (extra
         cysteine) hit, regardless of count.
      2. For whatever survives stage 1: reject if the count of the OTHER
         High-severity rules (DeAmdH, FragH, Isom) exceeds n_high, OR the
         Medium-severity count exceeds n_medium, OR the Low-severity count
         exceeds n_low.
    n_high/n_medium/n_low have no default -- pick explicit thresholds rather
    than silently assume one.
    """
    result = scan_lap_motifs_batch(sequences, cdr_only_input=True, rules=rules, include_extra_cysteine=True)
    has_ngly = result["per_rule_hits"]["Ngly"]
    has_xcys = result["per_rule_hits"]["xCys"]
    high_excl = high_count_excluding_ngly_xcys(result)

    keep = ~has_ngly & ~has_xcys
    keep &= high_excl <= n_high
    keep &= result["medium_count"] <= n_medium
    keep &= result["low_count"] <= n_low
    return keep


# ============================================================
# Annotator
# ============================================================

class AntibodyMotifAnnotator:
    """
    - Extracts per-chain regions (IMGT FR/CDR) using abnumber
    - Scans motifs in either:
        * CDR-only regions (cdrl* / cdrh*), OR
        * all regions (fw + cdr)
    - Adds extra cysteine flag (xCys) for any C found in CDRs (High)
    - Batch output:
        * per sequence: High/Medium/Low counts
        * if any High risks: include exact locations + flag (minimal decision info)
    """

    def __init__(
        self,
        extractor: Optional[IMGTRegionExtractor] = None,
        rules: Optional[List[MotifRule]] = None,
    ) -> None:
        self.extractor = extractor or IMGTRegionExtractor()
        self.rules = rules or build_rules()

    def annotate_one(
        self,
        seq_id: str,
        light_seq: str,
        heavy_seq: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Returns:
          detailed_df: one row per hit (ONLY hits; no empty rows)
          summary_row: minimal info dict for this sequence
        """
        L_raw = light_seq
        H_raw = heavy_seq
        L = self._clean_seq(light_seq)
        H = self._clean_seq(heavy_seq)

        l_regions = self.extractor.extract(L, "L")
        h_regions = self.extractor.extract(H, "H")

        hits: List[Dict[str, Any]] = []

        for chain_label, chain_seq, chain_regions in [
            ("L", L, l_regions),
            ("H", H, h_regions),
        ]:
            # -----------------------
            # Extra Cysteine in CDRs (High risk): any 'C' in CDRs
            # -----------------------
            cdr_regions = self._regions_for_scope(chain_regions, "cdr")
            for region in cdr_regions:
                s0, e0 = region.slice0()
                sub = chain_seq[s0:e0]
                for i, aa in enumerate(sub):
                    if aa == "C":
                        pos1 = region.start + i
                        hits.append(
                            {
                                "SequenceID": seq_id,
                                "Chain": chain_label,
                                "Region": region.name,
                                "Name": "Extra cysteine (CDRs)",
                                "Flag": "xCys",
                                "Severity": "High",
                                "Match": "C",
                                "HitDetail": f"C@{chain_label}{pos1}-{pos1}",
                                "Start": pos1,
                                "End": pos1,
                            }
                        )

            # -----------------------
            # Regex motif rules
            # -----------------------
            for rule in self.rules:
                regions_to_scan = self._regions_for_scope(chain_regions, rule.scope)

                for region in regions_to_scan:
                    s0, e0 = region.slice0()
                    sub = chain_seq[s0:e0]
                    overlap_pattern = re.compile(f"(?=({rule.pattern.pattern}))")

                    for m in overlap_pattern.finditer(sub):
                        match = m.group(1)
                        if not match:
                            continue
                    
                        start1 = region.start + m.start()
                        end1 = start1 + len(match) - 1
                    
                        hits.append({
                            "SequenceID": seq_id,
                            "Chain": chain_label,
                            "Region": region.name,
                            "Name": rule.name,
                            "Flag": rule.flag,
                            "Severity": rule.severity,
                            "Match": match,
                            "HitDetail": f"{match}@{chain_label}{start1}-{end1}",
                            "Start": start1,
                            "End": end1,
                        })

        detailed_df = pd.DataFrame(hits)

        # Summary: counts + high-risk details
        if detailed_df.empty:
            high_count = med_count = low_count = 0
            high_details = ""
        else:
            high_count = int((detailed_df["Severity"] == "High").sum())
            med_count = int((detailed_df["Severity"] == "Medium").sum())
            low_count = int((detailed_df["Severity"] == "Low").sum())

            if high_count > 0:
                high_df = detailed_df[detailed_df["Severity"] == "High"]
                high_details = "; ".join([f"{r.Flag}:{r.HitDetail}" for r in high_df.itertuples(index=False)])
            else:
                high_details = ""

        summary_row = {
            "SequenceID": seq_id,
            "light_chain": L_raw,
            "heavy_chain": H_raw,
            "HighRiskCount": high_count,
            "MediumRiskCount": med_count,
            "LowRiskCount": low_count,
            "HighRiskDetails": high_details,
        }

        return detailed_df, summary_row

    def batch_from_lc_hc_fasta(
        self,
        fasta_path: str,
        output_csv_path: str,
    ) -> pd.DataFrame:
        """
        Input FASTA:
          >id1
          LCSEQ:HCSEQ
          >id2
          LCSEQ:HCSEQ

        Output CSV/DF:
          - one row per SequenceID with:
              light_chain, heavy_chain, HighRiskCount, MediumRiskCount, LowRiskCount, HighRiskDetails
        """
        pairs = fasta_lc_hc_to_dict(fasta_path)

        summary_rows: List[Dict[str, Any]] = []

        for seq_id, (lc, hc) in pairs.items():
            try:
                _, summary = self.annotate_one(seq_id, lc, hc)
                summary_rows.append(summary)
                print(f"Done with {seq_id}")
            except Exception as e:
                summary_rows.append(
                    {
                        "SequenceID": seq_id,
                        "light_chain": lc,
                        "heavy_chain": hc,
                        "HighRiskCount": "",
                        "MediumRiskCount": "",
                        "LowRiskCount": "",
                        "HighRiskDetails": f"ERROR: {e}",
                    }
                )

        out_df = pd.DataFrame(summary_rows)
        out_df.to_csv(output_csv_path, index=False)
        return out_df

    # -----------------------
    # Internals
    # -----------------------

    @staticmethod
    def _clean_seq(seq: str) -> str:
        return re.sub(r"[^A-Z]", "", (seq or "").strip().upper())

    @staticmethod
    def _regions_for_scope(chain_regions: ChainRegions, scope: RegionScope) -> List[Region]:
        if scope == "all":
            return chain_regions.regions
        prefix = "cdrl" if chain_regions.chain == "L" else "cdrh"
        return chain_regions.by_prefix(prefix)


# # ============================================================
# # Example usage
# # ============================================================
# if __name__ == "__main__":
#     annotator = AntibodyMotifAnnotator()

#     # Single example (replace with your sequences)
#     lc = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLAWYQQKPGKAPKLLIYDASTRATGIPDRFSGSGSGTDFTLTISSLEPEDFAVYYCQQYNSYPWTFGQGTKVEIK"
#     hc = "EVQLVESGGGLVKPGGSLKVSCAASGFTFSSYAMSWVLSWVRQTPEKRLEWVATISWNSGSTYYPDSVKGRFTISRDNAKNTLYLQMSSLRSEDTAMYYCARDDGHTYYYGMDVWGAGTTVTVSS"

#     detailed, summary = annotator.annotate_one("example1", lc, hc)
#     print(summary)
#     print(detailed.head(25).to_string(index=False))

#     # Batch:
#     # df = annotator.batch_from_lc_hc_fasta("pairs.fasta", "risk_summary.csv")
#     # print(df.head().to_string(index=False))
