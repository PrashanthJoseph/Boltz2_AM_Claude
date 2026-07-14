from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

import LAP_inhouse as lap
import hydrophobic_analysis as hydro
import build_universe_bitmap as bub

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# Relative preference across the three move types when proposing children.
# Added mutations are more often deleterious than neutral/beneficial at an
# already-affinity-matured position, so lateral (same load, different edit)
# is preferred a little over increase/decrease rather than uniform 1/3 each.
DEFAULT_MOVE_WEIGHTS = {"increase": 0.3, "decrease": 0.3, "lateral": 0.4}


# ============================================================
# Config
# ============================================================

@dataclass
class SearchConfig:
    root_sequence: str
    designable_positions: Tuple[int, ...]  # 0-based indices into root_sequence

    # score_sign: the acquisition score always maximizes `score_sign * S(x)`.
    # +1 (default) assumes higher S(x) is better (e.g. pIC50-like, affinity-like).
    # Set -1 if your Boltz-2 wrapper reports lower = better (e.g. ddG-/energy-like).
    # CONFIRM THIS against the actual Boltz-2 reward wrapper before trusting results.
    score_sign: int = 1

    # No mutation-load weight in ranking (removed) -- discouraging bold jumps
    # lives entirely in combined_badness/kappa (jump sizing), not in a
    # standing ranking penalty a child would carry forever. See run_search.
    beta: float = 0.5          # exploration weight (static -- U(x) already decays
                                # organically as the pool fills in nearby, so a
                                # separate time-based anneal was redundant)
    # U(x) = novelty = nearest-neighbor Hamming distance / n_design (no radius
    # parameter needed -- "how far is the closest thing already tried")

    j_base: int = 1
    j_max: int = 3
    # kappa: scales combined performance-badness into extra jump size. Jump
    # size is now driven by raw S(x) trajectory (is this parent underperforming
    # its own parent, and/or its same-distance-from-root peers?), NOT novelty
    # (U stays in base_score for parent-selection ranking, but no longer sets
    # jump size). combined_badness sums both MAD-normalized deltas (bad+bad
    # compounds, good can offset bad), floored at 0 (never negative jump).
    # jump = clip(round(j_base + kappa*combined_badness), j_base, j_max).
    kappa: float = 1.0

    # Relative preference across increase/decrease/lateral moves. Default
    # favors lateral slightly, since added mutations are more often
    # deleterious than a same-load alternative edit at an already
    # affinity-matured position. Renormalized over whichever moves are
    # feasible for a given parent (see propose_children).
    move_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MOVE_WEIGHTS))

    k_parents: int = 8
    k_children_per_parent: int = 8
    # Child proposal enumerates the FULL reachable neighborhood per move type
    # and safety-checks all of it up front (see propose_safe_children /
    # enumerate_neighborhood) -- true selection from a known-safe population,
    # not generate-a-guess-and-retry, so there's no oversample/retry knob here.

    patience_generations: int = 8
    max_boltz2_calls: int = 3000

    elitism: bool = True

    # fitness_gate_tau: quality gate on parent ELIGIBILITY. A member is
    # parent-eligible only if its normalized_objective is no more than tau
    # MADs BELOW the pool MEDIAN; everything further behind is excluded from
    # selection REGARDLESS of how novel it is. This caps the "novelty rescue"
    # ceiling (novelty can lift base_score by at most beta*(U_max - U_min)
    # ~= 0.43 MAD, so without a gate an isolated mediocre member can
    # persistently hold a parent slot in a sparse space).
    #
    # Reference is the MEDIAN, not the best: normalized_objective is already
    # median-centered ((obj-med)/MAD), so the gate is simply "z >= -tau". This
    # is deliberately robust to a runaway-best outlier -- a best-relative gate
    # (AdaLead's mu*max_score analog) collapses to elite-only when one early
    # hit sits many MADs above the pack, throttling exploration; the median is
    # stable under that. The gate still excludes genuinely-bad members (well
    # below typical) but never prunes members sitting near the pool's middle.
    #
    # None (default) = OFF, preserving prior behavior. The elite is always
    # exempt (elitism re-adds it below), so the gate can never strand the
    # best-found member. Sensible values are ~2-3 (MAD is scaled to std-like
    # units, so tau reads as "how many sigma below typical is still allowed").
    fitness_gate_tau: Optional[float] = None

    # Safety policy (LAP severities, hydrophobic patch length) is NOT here --
    # it lives entirely in whichever SafetyChecker object is passed to
    # run_search(), so it stays in one place instead of two configs that
    # could silently disagree.


@dataclass
class PoolMember:
    design: str          # the 7-character designable substring
    score: float          # raw S(x) from the oracle, in its native convention
    generation: int
    parent_index: Optional[int]
    step: int             # Hamming distance to immediate parent (0 for root)
    move_type: Optional[str] = None  # "increase" / "decrease" / "lateral" / None for root


# ============================================================
# Sequence assembly / distance helpers
# ============================================================

def assemble_full_sequence(root_sequence: str, designable_positions: Sequence[int], design: str) -> str:
    chars = list(root_sequence)
    for pos, ch in zip(designable_positions, design):
        chars[pos] = ch
    return "".join(chars)


def initial_design_string(root_sequence: str, designable_positions: Sequence[int]) -> str:
    return "".join(root_sequence[p] for p in designable_positions)


def hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def robust_mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)) * 1.4826 + 1e-9)


def propose_children(
    parent_design: str,
    root_design: str,
    jump: int,
    n_proposals: int,
    rng: np.random.Generator,
    move_weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, str]]:
    """
    Proposes candidate children via three deliberate move types, chosen
    explicitly per proposal rather than left to chance:
      - increase: mutate `jump` positions currently equal to root, away from
        root -> moves to a HIGHER mutational-load bracket.
      - decrease: revert `jump` currently-mutated positions back to root's
        exact residue -> moves to a LOWER mutational-load bracket, on purpose.
      - lateral: swap `jump` currently-mutated positions to a *different*
        non-root residue -> same mutational load, tests an alternative edit.
    Returns (candidate_design, move_type) pairs.

    move_weights (default DEFAULT_MOVE_WEIGHTS): relative preference across
    the three move types, since added mutations are more often deleterious
    than neutral/beneficial edits at an already-affinity-matured position --
    biases toward lateral > increase/decrease rather than uniform 1/3 each.
    Renormalized over whichever moves are actually feasible for this parent
    (e.g. root has no mutated positions yet, so it's "increase" regardless
    of these weights).
    """
    weights = move_weights if move_weights is not None else DEFAULT_MOVE_WEIGHTS

    n = len(parent_design)
    mutated_idx = [i for i in range(n) if parent_design[i] != root_design[i]]
    unmutated_idx = [i for i in range(n) if parent_design[i] == root_design[i]]

    feasible = []
    if unmutated_idx:
        feasible.append("increase")
    if mutated_idx:
        feasible.append("decrease")
        feasible.append("lateral")
    if not feasible:
        return []

    feasible_weights = np.array([weights[m] for m in feasible], dtype=np.float64)
    feasible_weights /= feasible_weights.sum()

    out: List[Tuple[str, str]] = []
    for _ in range(n_proposals):
        move = rng.choice(feasible, p=feasible_weights)
        candidate = list(parent_design)

        if move == "increase":
            k = min(jump, len(unmutated_idx))
            for p in rng.choice(unmutated_idx, size=k, replace=False):
                choices = [a for a in AA_ALPHABET if a != root_design[p]]
                candidate[p] = rng.choice(choices)
        elif move == "decrease":
            k = min(jump, len(mutated_idx))
            for p in rng.choice(mutated_idx, size=k, replace=False):
                candidate[p] = root_design[p]
        else:  # lateral
            k = min(jump, len(mutated_idx))
            for p in rng.choice(mutated_idx, size=k, replace=False):
                choices = [a for a in AA_ALPHABET if a != root_design[p] and a != candidate[p]]
                if not choices:
                    continue
                candidate[p] = rng.choice(choices)

        out.append(("".join(candidate), move))
    return out


def enumerate_neighborhood(parent_design: str, root_design: str, move: str, jump: int) -> List[str]:
    """
    Fully enumerates EVERY candidate reachable from parent_design via exactly
    this move type and jump size -- the complete combinatorial neighborhood,
    no randomness. Worst case (move="increase", jump=3, all 7 positions
    eligible) is C(7,3)*19^3 = 240,065 candidates -- trivial to generate and
    vectorized-safety-check in one shot, which is what makes true selection
    (rather than generate-a-guess-and-retry) actually cheap enough to do.
    """
    n = len(parent_design)
    mutated_idx = [i for i in range(n) if parent_design[i] != root_design[i]]
    unmutated_idx = [i for i in range(n) if parent_design[i] == root_design[i]]
    pool_idx = unmutated_idx if move == "increase" else mutated_idx
    if len(pool_idx) < jump:
        return []

    out: List[str] = []
    for positions in itertools.combinations(pool_idx, jump):
        if move == "decrease":
            candidate = list(parent_design)
            for p in positions:
                candidate[p] = root_design[p]
            out.append("".join(candidate))
            continue

        if move == "increase":
            choice_lists = [[a for a in AA_ALPHABET if a != root_design[p]] for p in positions]
        else:  # lateral
            choice_lists = [
                [a for a in AA_ALPHABET if a != root_design[p] and a != parent_design[p]] for p in positions
            ]
        for combo in itertools.product(*choice_lists):
            candidate = list(parent_design)
            for p, a in zip(positions, combo):
                candidate[p] = a
            out.append("".join(candidate))
    return out


def propose_safe_children(
    parent_design: str,
    root_design: str,
    jump: int,
    target_n: int,
    already_seen: set,
    safety_checker: "SafetyChecker",
    rng: np.random.Generator,
    move_weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, str]]:
    """
    Real selection, not rejection sampling: for each feasible move type,
    enumerate the ENTIRE reachable neighborhood at this jump size, safety-check
    all of it in one vectorized call, then sample target_n from the resulting
    known-safe population -- proportioned across move types by move_weights,
    with any shortfall (a move type running out of safe candidates) filled
    from whichever other feasible move type still has some left, in weight
    order. Every candidate considered here is already guaranteed safe before
    it's ever picked, since the whole neighborhood was checked up front.
    """
    weights = move_weights if move_weights is not None else DEFAULT_MOVE_WEIGHTS
    n = len(parent_design)
    mutated_idx = [i for i in range(n) if parent_design[i] != root_design[i]]
    unmutated_idx = [i for i in range(n) if parent_design[i] == root_design[i]]

    feasible = []
    if unmutated_idx:
        feasible.append("increase")
    if mutated_idx:
        feasible.append("decrease")
        feasible.append("lateral")
    if not feasible:
        return []

    safe_by_move: Dict[str, List[str]] = {}
    for move in feasible:
        candidates = enumerate_neighborhood(parent_design, root_design, move, jump)
        fresh = [c for c in candidates if c not in already_seen]
        if fresh:
            safe_mask = safety_checker.is_safe(fresh)
            safe = [c for c, ok in zip(fresh, safe_mask) if ok]
        else:
            safe = []
        rng.shuffle(safe)
        safe_by_move[move] = safe

    # move_weights is a per-child PROBABILITY, not a batch-level target to
    # hit exactly: draw each of target_n children's move type independently
    # via a weighted coin flip, same as the original stochastic design --
    # the stated ratio only emerges as an expectation over many draws, not a
    # deterministic count within one small batch. Safe pools were already
    # enumerated + shuffled above, so each draw is just "next unused
    # candidate of the drawn type" -- still zero rejection sampling, since
    # every candidate offered was already confirmed safe up front.
    cursors = {m: 0 for m in feasible}
    accepted: List[Tuple[str, str]] = []
    local_seen: set = set()
    for _ in range(target_n):
        available = [m for m in feasible if cursors[m] < len(safe_by_move[m])]
        if not available:
            break  # every move type's safe pool is exhausted
        avail_weights = np.array([weights[m] for m in available], dtype=np.float64)
        avail_weights /= avail_weights.sum()
        move = rng.choice(available, p=avail_weights)
        cand = safe_by_move[move][cursors[move]]
        cursors[move] += 1
        if cand in local_seen:
            continue
        local_seen.add(cand)
        accepted.append((cand, move))

    return accepted[:target_n]


# ============================================================
# Candidate safety checking (LAP + hydrophobicity), pluggable
# ============================================================

class SafetyChecker(Protocol):
    def is_safe(self, design_substrings: Sequence[str]) -> np.ndarray: ...


class LiveFilterSafetyChecker:
    """Runs the real vectorized LAP + hydrophobicity filters on demand.
    Correct for any pool size, but does real work every call — fine at the
    small per-generation batch sizes this search proposes."""

    def __init__(
        self,
        root_sequence: str,
        designable_positions: Sequence[int],
        n_high: int,
        n_medium: int,
        n_low: int,
        max_allowed_patch_len: int = hydro.MIN_PATCH_LEN,
    ) -> None:
        # n_high/n_medium/n_low: max allowed count of non-Ngly/xCys High,
        # Medium, and Low severity hits respectively -- Ngly/xCys are always
        # hard-rejected regardless of these. See LAP_inhouse.passes_lap_policy.
        # max_allowed_patch_len: rejection ceiling on longest hydrophobic
        # patch -- lower it for stricter rejection of shorter patches.
        self.root_sequence = root_sequence
        self.designable_positions = designable_positions
        self.n_high = n_high
        self.n_medium = n_medium
        self.n_low = n_low
        self.max_allowed_patch_len = max_allowed_patch_len

    def is_safe(self, design_substrings: Sequence[str]) -> np.ndarray:
        if not design_substrings:
            return np.zeros(0, dtype=bool)
        full_seqs = [
            assemble_full_sequence(self.root_sequence, self.designable_positions, d)
            for d in design_substrings
        ]
        lap_ok = lap.passes_lap_policy(
            full_seqs, n_high=self.n_high, n_medium=self.n_medium, n_low=self.n_low
        )
        hres = hydro.analyze_hydrophobicity_batch(
            full_seqs, cdr_only_input=True, max_allowed_patch_len=self.max_allowed_patch_len
        )
        return lap_ok & hres["passes_patch_filter"]


class BitmapSafetyChecker:
    """
    Safety check against the precomputed full-universe facts (see
    build_universe_bitmap.build_universe_facts / load_facts_memmap) instead
    of re-scanning LAP/hydrophobicity live every call. Deliberately SPARSE:
    each is_safe() call encodes the (small, per-generation) batch of
    candidates to their integer indices and fancy-indexes just those
    positions out of the memmapped fact arrays -- never materializes or
    touches the full ~5-7GB of facts, so this stays cheap even when the
    facts don't fit in RAM (that's the whole point of memmap here).
    """

    def __init__(
        self,
        facts: dict,
        n_high: int,
        n_medium: int,
        n_low: int,
        max_allowed_patch_len: int,
        alphabet: str = AA_ALPHABET,
    ):
        self.facts = facts
        self.n_high = n_high
        self.n_medium = n_medium
        self.n_low = n_low
        self.max_allowed_patch_len = max_allowed_patch_len
        self.alphabet = alphabet
        self.char_to_idx = {c: i for i, c in enumerate(alphabet)}

    def encode(self, design: str) -> int:
        idx = 0
        base = len(self.alphabet)
        for power, ch in enumerate(design):
            idx += self.char_to_idx[ch] * (base ** power)
        return idx

    def is_safe(self, design_substrings: Sequence[str]) -> np.ndarray:
        if not design_substrings:
            return np.zeros(0, dtype=bool)
        indices = np.array([self.encode(d) for d in design_substrings], dtype=np.int64)
        return bub.apply_policy(
            np.asarray(self.facts["has_ngly"][indices]),
            np.asarray(self.facts["has_xcys"][indices]),
            np.asarray(self.facts["high_count_excl"][indices]),
            np.asarray(self.facts["medium_count"][indices]),
            np.asarray(self.facts["low_count"][indices]),
            np.asarray(self.facts["max_patch_len"][indices]),
            n_high=self.n_high, n_medium=self.n_medium, n_low=self.n_low,
            max_allowed_patch_len=self.max_allowed_patch_len,
        )


# ============================================================
# Main search loop
# ============================================================

def run_search(
    config: SearchConfig,
    score_fn: Callable[[Sequence[str]], np.ndarray],
    safety_checker: SafetyChecker,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    rng = rng or np.random.default_rng()
    design_positions = config.designable_positions
    n_design = len(design_positions)

    pool: List[PoolMember] = []
    design_strings: List[str] = []
    seen = set()

    def add_member(design: str, score: float, generation: int, parent_index: Optional[int],
                   step: int, move_type: Optional[str] = None) -> None:
        pool.append(PoolMember(design=design, score=score, generation=generation,
                                parent_index=parent_index, step=step, move_type=move_type))
        design_strings.append(design)
        seen.add(design)

    root_design = initial_design_string(config.root_sequence, design_positions)
    root_full = assemble_full_sequence(config.root_sequence, design_positions, root_design)
    root_score = float(score_fn([root_full])[0])
    add_member(root_design, root_score, generation=0, parent_index=None, step=0)

    calls_made = 1
    generations_without_improve = 0
    generation = 0
    log: List[Dict] = []

    while calls_made < config.max_boltz2_calls and generations_without_improve < config.patience_generations:
        generation += 1

        arr = hydro.strings_to_char_array(design_strings, n_design)
        scores = np.array([m.score for m in pool])
        objective = config.score_sign * scores  # higher is always better after this line

        # U(x): nearest-neighbor Hamming distance, normalized to [0,1]. No
        # radius parameter -- "how far is the single closest thing already
        # tried" rather than "how many things are within an arbitrary cutoff."
        dist_matrix = (arr[:, None, :] != arr[None, :, :]).sum(axis=2)
        np.fill_diagonal(dist_matrix, n_design + 1)  # exclude self from its own nearest-neighbor search
        nearest_dist = dist_matrix.min(axis=1)
        U = nearest_dist / n_design

        # Robust (median/MAD) normalization of the objective for RANKING only.
        # beta is calibrated assuming an O(1)-scale objective; Boltz-2's raw
        # units (ddG-like, pIC50-like, whatever) are unknown a priori, so
        # normalize against the pool's own observed spread each generation
        # instead of guessing the raw scale. Elite selection and best-score
        # tracking below still use the raw `objective`, since that should
        # stay stable across generations rather than drift with the
        # normalization.
        med = np.median(objective)
        mad = robust_mad(objective)
        normalized_objective = (objective - med) / mad

        # No step/mutation-load term here on purpose: a node's ranking is
        # based purely on its own performance + novelty, never on how big
        # the edit was that created it -- that would permanently penalize a
        # genuinely good result for having arrived via a bold jump, forever,
        # every generation it's considered as a future parent. Discouraging
        # bold jumps lives entirely in combined_badness/kappa below, applied
        # once, at the moment a parent's jump size is decided -- not baked
        # into the resulting child's standing ranking.
        base_score = normalized_objective + config.beta * U

        # Performance-based jump sizing (independent of U/novelty above, which
        # stays confined to ranking/selection). Two questions per pool member:
        # (1) was the edit that created it an improvement over its own parent
        # (root counts as its own parent -> delta 0), and (2) is it doing
        # better or worse than its same-distance-from-root peers. Both are
        # raw-objective deltas (the actual Boltz-2 optimizing metric), not the
        # adjusted base_score used for ranking.
        parent_obj = np.array([
            objective[m.parent_index] if m.parent_index is not None else objective[i]
            for i, m in enumerate(pool)
        ])
        parent_delta = objective - parent_obj

        dist_from_root = np.array([hamming(d, root_design) for d in design_strings])
        bracket_delta = np.zeros(len(pool))
        for b in np.unique(dist_from_root):
            idx_b = np.where(dist_from_root == b)[0]
            if len(idx_b) <= 1:
                continue  # alone in this bracket (e.g. root) -> stays 0, no peers to compare to
            for i in idx_b:
                others = objective[idx_b[idx_b != i]]
                bracket_delta[i] = objective[i] - np.median(others)

        # Normalize each delta by the spread of ITS OWN kind across the pool
        # (typical size of "one edit's effect", typical size of "peer spread
        # at a given bracket") rather than the whole pool's objective MAD --
        # that mismatched an edit-sized signal against a whole-campaign-sized
        # scale, which is why jump size barely moved off the floor in testing.
        mad_parent_delta = robust_mad(parent_delta)
        mad_bracket_delta = robust_mad(bracket_delta)
        combined_badness = np.maximum(
            0.0, -((parent_delta / mad_parent_delta) + (bracket_delta / mad_bracket_delta))
        )

        order = np.argsort(-base_score)  # all pool indices, descending by base_score

        elite_idx = int(np.argmax(objective))

        # Fitness gate (optional quality floor): a member is parent-eligible
        # only if its normalized_objective is no more than tau MADs below the
        # pool median. normalized_objective is already median-centered and in
        # MAD units, so the gate is just "z >= -tau". This runs on RAW fitness
        # (no novelty term) on purpose -- the whole point is to stop novelty
        # from keeping a low-scoring member eligible. The elite is exempt
        # (re-added via elitism below), so the gate can never strand the
        # best-found member even if numerical edge cases would exclude it.
        if config.fitness_gate_tau is not None:
            eligible_mask = normalized_objective >= -config.fitness_gate_tau
            eligible_mask[elite_idx] = True
        else:
            eligible_mask = np.ones(len(pool), dtype=bool)
        n_gated_out = int((~eligible_mask).sum())

        selected: List[int] = []
        if config.elitism:
            selected.append(elite_idx)
        for i in order.tolist():
            if len(selected) >= config.k_parents:
                break
            if not eligible_mask[i]:
                continue
            if i not in selected:
                selected.append(i)

        new_records: List[Tuple[str, int, int, str]] = []  # (design, parent_index, step, move_type)
        jumps_used: Dict[int, int] = {}

        for parent_idx in selected:
            parent_design = design_strings[parent_idx]
            jump = int(np.clip(round(config.j_base + config.kappa * combined_badness[parent_idx]),
                                config.j_base, config.j_max))
            jumps_used[parent_idx] = jump

            accepted = propose_safe_children(
                parent_design, root_design, jump, config.k_children_per_parent,
                seen, safety_checker, rng, move_weights=config.move_weights,
            )

            for cand, mv in accepted:
                new_records.append((cand, parent_idx, hamming(cand, parent_design), mv))
                seen.add(cand)

        if not new_records:
            generations_without_improve += 1
            log.append({"generation": generation, "n_new": 0, "best_objective": float(objective.max()),
                         "calls_made": calls_made, "beta": config.beta})
            continue

        full_seqs = [assemble_full_sequence(config.root_sequence, design_positions, d) for d, _, _, _ in new_records]
        new_scores = score_fn(full_seqs)
        calls_made += len(full_seqs)

        prev_best = float(objective.max())
        for (design, parent_idx, step, mv), s in zip(new_records, new_scores):
            add_member(design, float(s), generation, parent_idx, step, mv)

        cur_best = float((config.score_sign * np.array([m.score for m in pool])).max())
        generations_without_improve = 0 if cur_best > prev_best + 1e-9 else generations_without_improve + 1

        move_counts = {"increase": 0, "decrease": 0, "lateral": 0}
        for _, _, _, mv in new_records:
            move_counts[mv] += 1

        log.append({
            "generation": generation,
            "n_new": len(new_records),
            "calls_made": calls_made,
            "best_objective": cur_best,
            "beta": config.beta,
            "generations_without_improve": generations_without_improve,
            "move_counts": move_counts,
            "jumps_used": jumps_used,
            "elite_combined_badness": float(combined_badness[elite_idx]),
            "n_gated_out": n_gated_out,
        })

    return {"pool": pool, "log": log, "calls_made": calls_made}
