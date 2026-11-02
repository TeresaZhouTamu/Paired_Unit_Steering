#!/usr/bin/env python3
"""Score trajectories from any optimizer against independent local directions.

For each sampled labeled reference pool: match positive/negative proteins,
select directions near each trajectory's starting protein, and score each
realized representation change. The trajectory score is its lower quartile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


EPS = 1e-8
TRAJECTORY_COLUMNS = (
    "sample_id", "method", "step", "original_sequence",
    "before_sequence", "after_sequence",
)


# 1. Load and validate an independent reference set and realized paths.
def clean_sequence(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Sequences must be nonempty strings")
    value = value.strip().upper()
    if not value.isascii() or not value.isalpha():
        raise ValueError(f"Sequence contains nonletter characters: {value[:40]!r}")
    return value


def select_property(frame: pd.DataFrame, property_name: str | None, path: Path) -> pd.DataFrame:
    """Select one property before target IDs are matched across input files."""
    if "property" not in frame:
        return frame
    if property_name is None:
        raise ValueError(f"{path} has a property column; specify --property")
    selected = frame.loc[frame["property"].astype(str) == property_name].copy()
    if selected.empty:
        raise ValueError(f"{path} has no rows for property {property_name!r}")
    return selected


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), EPS)


def expand_mutation_logs(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept the optimized-sequence/mutation-log format used by PU and ASPO."""
    columns = {"sample_id", "method", "original_sequence", "sequence", "mutation_log"}
    if not columns.issubset(frame.columns):
        raise ValueError(f"Mutation-log CSV needs columns: {sorted(columns)}")
    if frame.duplicated(["sample_id", "method"]).any():
        raise ValueError("Only one mutation log per sample_id/method is allowed")
    records = []
    for row in frame.itertuples(index=False):
        original = clean_sequence(row.original_sequence)
        expected = clean_sequence(row.sequence)
        if not isinstance(row.mutation_log, str):
            raise ValueError(f"{row.sample_id}/{row.method}: mutation_log must be JSON")
        try:
            changes = json.loads(row.mutation_log)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{row.sample_id}/{row.method}: invalid mutation_log JSON") from exc
        if not isinstance(changes, list):
            raise ValueError("Each mutation_log must be a JSON list")
        by_round: dict[int, list[dict]] = {}
        for change in changes:
            if not isinstance(change, dict):
                raise ValueError("Each mutation must be a JSON object")
            round_id = int(change.get("round", change.get("step", 1)))
            if round_id < 1:
                raise ValueError("Mutation round must be positive")
            by_round.setdefault(round_id, []).append(change)
        chars = list(original)
        for round_id in sorted(by_round):
            before = "".join(chars)
            used_positions = set()
            for change in by_round[round_id]:
                pos = next((change[k] for k in (
                    "residue_position_1based", "token_position", "position"
                ) if k in change), None)
                if pos is None:
                    raise ValueError("Mutation lacks a 1-based residue/token position")
                idx = int(pos) - 1
                old, new = clean_sequence(change.get("old")), clean_sequence(change.get("new"))
                if len(old) != 1 or len(new) != 1 or idx < 0 or idx >= len(chars):
                    raise ValueError("Mutation must change one valid residue position")
                if idx in used_positions or chars[idx] != old:
                    raise ValueError(f"{row.sample_id}/{row.method}: conflicting mutation at {pos}")
                used_positions.add(idx)
                chars[idx] = new
            records.append({"sample_id": row.sample_id, "method": row.method,
                            "step": round_id, "original_sequence": original,
                            "before_sequence": before, "after_sequence": "".join(chars)})
        if "".join(chars) != expected:
            raise ValueError(f"{row.sample_id}/{row.method}: reconstructed endpoint mismatch")
    if not records:
        raise ValueError("No mutation rounds found; LDS needs realized steps")
    return pd.DataFrame(records)


def read_trajectories(path: Path, property_name: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"sample_id": str, "method": str})
    if df.empty:
        raise ValueError("Trajectory CSV is empty")
    df = select_property(df, property_name, path)
    if not set(TRAJECTORY_COLUMNS).issubset(df.columns):
        df = expand_mutation_logs(df)
    df = df[list(TRAJECTORY_COLUMNS)].copy()
    for col in ("original_sequence", "before_sequence", "after_sequence"):
        df[col] = df[col].map(clean_sequence)
    if df[["sample_id", "method"]].isna().any().any():
        raise ValueError("sample_id and method must be present")
    df["step"] = pd.to_numeric(df.step, errors="raise")
    if df.step.isna().any() or not (df.step == np.floor(df.step)).all():
        raise ValueError("step must contain integer round indices")
    df["step"] = df.step.astype(int)
    if (df.step < 1).any() or df.duplicated(["sample_id", "method", "step"]).any():
        raise ValueError("Steps must be positive and unique within each target/method")
    df = df.sort_values(["sample_id", "method", "step"], kind="stable").reset_index(drop=True)
    for (sid, method), group in df.groupby(["sample_id", "method"], sort=False):
        original = group.original_sequence.iloc[0]
        if group.original_sequence.nunique() != 1:
            raise ValueError(f"{sid}/{method}: inconsistent starting protein")
        if group.before_sequence.iloc[0] != original:
            raise ValueError(f"{sid}/{method}: first before_sequence differs from original")
        before = group.before_sequence.tolist()[1:]
        previous = group.after_sequence.tolist()[:-1]
        if before != previous:
            raise ValueError(f"{sid}/{method}: consecutive steps do not form a trajectory")
    return df


def read_trajectory_files(paths: list[Path], property_name: str | None = None) -> pd.DataFrame:
    """Combine complete per-method runs; a candidate may occur in only one file."""
    frames = [read_trajectories(path, property_name) for path in paths]
    for path, frame in zip(paths, frames):
        frame["source_file"] = str(path)
    combined = pd.concat(frames, ignore_index=True)
    sources_per_candidate = combined.groupby(["sample_id", "method"]).source_file.nunique()
    if (sources_per_candidate > 1).any():
        raise ValueError("A sample_id/method pair appears in multiple trajectory files")
    if combined.duplicated(["sample_id", "method", "step"]).any():
        raise ValueError("Duplicate trajectory step across input files")
    return combined


def read_reference(path: Path, positive: str, negative: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"label": str})
    if not {"sequence", "label"}.issubset(df.columns):
        raise ValueError("Reference CSV needs 'sequence' and 'label' columns")
    df = df[["sequence", "label"]].copy()
    df.sequence = df.sequence.map(clean_sequence)
    df.label = df.label.astype(str).str.strip()
    if positive == negative or set(df.label) - {positive, negative}:
        raise ValueError(f"Reference labels must be {negative!r} or {positive!r}")
    if df.groupby("sequence").label.nunique().gt(1).any():
        raise ValueError("Reference sequence has conflicting labels")
    return df.drop_duplicates("sequence").reset_index(drop=True)


def clean_reference(
    source: pd.DataFrame, trajectories: pd.DataFrame, extra_paths: list[Path]
) -> tuple[pd.DataFrame, dict[str, int]]:
    excluded = set()
    for col in ("original_sequence", "before_sequence", "after_sequence"):
        excluded.update(trajectories[col].tolist())
    for path in extra_paths:
        frame = pd.read_csv(path)
        if "sequence" not in frame:
            raise ValueError(f"{path} requires a 'sequence' column")
        excluded.update(frame.sequence.map(clean_sequence).tolist())
    kept = source[~source.sequence.isin(excluded)].copy().reset_index(drop=True)
    return kept, {"reference_unique": len(source), "excluded_exact_overlap": len(source)-len(kept),
                  "reference_after_exclusion": len(kept)}


# 2. Construct local property directions and measure step agreement.
def build_pair_bank(
    positive_pair: np.ndarray, negative_pair: np.ndarray,
    positive_direction: np.ndarray, negative_direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match by pairing-layer cosine; orient direction negative -> positive."""
    pp, npair = _unit(positive_pair), _unit(negative_pair)
    if min(len(pp), len(npair)) < 1:
        raise ValueError("Both labels must be present in the sampled pool")
    pi, ni = linear_sum_assignment(-(pp @ npair.T))
    # The midpoint uses the original embeddings; normalize only afterward.
    background = _unit((positive_pair[pi] + negative_pair[ni]) / 2)
    direction = _unit(positive_direction[pi] - negative_direction[ni])
    if (np.linalg.norm(background, axis=-1) <= EPS).any() or (
        np.linalg.norm(direction, axis=-1) <= EPS
    ).any():
        raise ValueError("A matched pair has undefined background or property direction")
    return background.astype(np.float32), direction.astype(np.float32)


def step_agreement(displacement: np.ndarray, local_directions: np.ndarray) -> float:
    if np.linalg.norm(displacement) <= EPS:
        return 0.5  # Same zero-step convention as the supplied notebooks.
    return float(np.mean(local_directions @ _unit(displacement) > 0))


class ESMCFeatures:
    """Lazy ESMC loading; cache only the two layers needed for LDS."""

    def __init__(self, model_name: str, device: str, pairing_layer: int, optimal_layer: int):
        try:
            import torch
            from esm.models.esmc import ESMC
            from esm.tokenization import get_esmc_model_tokenizers
        except ImportError as exc:
            raise RuntimeError(
                "Install torch and an esm release exposing the legacy ESMC API"
            ) from exc
        self.torch = torch
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto" else device
        )
        self.model = ESMC.from_pretrained(model_name, device=self.device)
        self.model = self.model.to(torch.float32).to(self.device).eval()
        self.tokenizer = get_esmc_model_tokenizers()
        self.pair_layer = pairing_layer
        self.optimal_layer = optimal_layer
        self.cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __call__(self, seq: str) -> tuple[np.ndarray, np.ndarray]:
        if seq not in self.cache:
            torch = self.torch
            raw = self.tokenizer.encode(seq, add_special_tokens=False)
            bos = getattr(self.tokenizer, "bos_token_id", 0) or 0
            eos = getattr(self.tokenizer, "eos_token_id", 2) or 2
            tokens = torch.tensor([bos, *raw, eos], device=self.device, dtype=torch.long)
            with torch.no_grad():
                output = self.model(
                    sequence_tokens=tokens.unsqueeze(0),
                    sequence_id=torch.ones_like(tokens).unsqueeze(0),
                )
            h = output.hidden_states
            if self.pair_layer >= len(h) or self.optimal_layer >= len(h):
                raise ValueError("Requested layer exceeds the model's hidden-state layers")
            pair = h[self.pair_layer, 0, 1:-1].mean(dim=0).cpu().numpy().astype(np.float32)
            optimal = h[self.optimal_layer, 0, 1:-1].mean(dim=0).cpu().numpy().astype(np.float32)
            self.cache[seq] = (pair, optimal)
        return self.cache[seq]


def parse_seeds(raw: str) -> list[int]:
    try:
        seeds = [int(v.strip()) for v in raw.split(",")]
    except ValueError as exc:
        raise ValueError("--seeds must be comma-separated integers") from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("At least one distinct pool seed is required")
    return seeds


# 3. Summarize trajectory support and prioritize candidates.
def rank_candidates(
    candidate_mean: pd.DataFrame, traces: pd.DataFrame, top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank candidates solely by mean Q25 LDS across independent pools."""
    endpoints = []
    for (sid, method), group in traces.groupby(["sample_id", "method"], sort=False):
        group = group.sort_values("step")
        sequence_path = [group.original_sequence.iloc[0], *group.after_sequence.tolist()]
        endpoints.append({
            "sample_id": sid, "method": method,
            "original_sequence": sequence_path[0],
            "sequence": sequence_path[-1],
            "trajectory_path": json.dumps(sequence_path),
        })
    result = candidate_mean.merge(
        pd.DataFrame(endpoints), on=["sample_id", "method"],
        validate="one_to_one",
    )
    if len(result) != len(candidate_mean):
        raise ValueError("Some LDS scores lack a matching trajectory endpoint")
    result = result.sort_values(
        ["lds_q25_mean_pools", "method", "sample_id"],
        ascending=[False, True, True], kind="stable",
    ).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result, result.head(top_n).copy()


def score_reference_pools(
    pool: pd.DataFrame, positive: pd.DataFrame, negative: pd.DataFrame,
    traces: pd.DataFrame, embed: ESMCFeatures, seeds: list[int],
    per_class: int, local_k: int,
) -> tuple[pd.DataFrame, list[dict]]:
    """Measure every realized step against each independently sampled pool."""
    # Cache each sequence embedding once across all pools and trajectories.
    for sequence in pd.unique(pd.concat([
        pool.sequence, traces.original_sequence, traces.before_sequence,
        traces.after_sequence,
    ], ignore_index=True)):
        embed(sequence)

    starts = {(sid, method): original for (sid, method), original in
              traces.groupby(["sample_id", "method"]).original_sequence.first().items()}
    displacements = [(
        row.sample_id, row.method, int(row.step),
        embed(row.after_sequence)[1] - embed(row.before_sequence)[1],
    ) for row in traces.itertuples(index=False)]

    step_rows, pool_details = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        pos_sequences = positive.iloc[
            rng.choice(len(positive), size=per_class, replace=False)
        ].sequence.tolist()
        neg_sequences = negative.iloc[
            rng.choice(len(negative), size=per_class, replace=False)
        ].sequence.tolist()
        backgrounds, directions = build_pair_bank(
            np.stack([embed(s)[0] for s in pos_sequences]),
            np.stack([embed(s)[0] for s in neg_sequences]),
            np.stack([embed(s)[1] for s in pos_sequences]),
            np.stack([embed(s)[1] for s in neg_sequences]),
        )
        effective_k = min(local_k, len(backgrounds))

        # For each candidate, choose K matched pairs nearest its starting background.
        local_directions = {}
        for key, original in starts.items():
            cosine = backgrounds @ _unit(embed(original)[0])
            indices = np.argsort(-cosine, kind="stable")[:effective_k]
            local_directions[key] = directions[indices]

        for sid, method, step, delta in displacements:
            step_rows.append({
                "pool_seed": seed, "sample_id": sid, "method": method,
                "step": step,
                "lds": step_agreement(delta, local_directions[sid, method]),
            })
        pool_details.append({
            "pool_seed": seed, "matched_pairs": len(backgrounds),
            "local_k_effective": effective_k,
        })
    return pd.DataFrame(step_rows), pool_details


def summarize_lds(step_frame: pd.DataFrame, traces: pd.DataFrame, top_n: int):
    """Take trajectory Q25, average over pools, then rank candidates."""
    trajectories = step_frame.groupby(
        ["pool_seed", "sample_id", "method"], as_index=False,
    ).agg(
        lds_q25=("lds", lambda x: float(np.quantile(x, 0.25))),
        lds_mean=("lds", "mean"), n_steps=("lds", "size"),
    )
    method_by_pool = trajectories.groupby(["pool_seed", "method"], as_index=False).agg(
        mean_lds_q25=("lds_q25", "mean"), n_targets=("sample_id", "nunique"),
    )
    candidate_mean = trajectories.groupby(["sample_id", "method"], as_index=False).agg(
        lds_q25_mean_pools=("lds_q25", "mean"),
        lds_q25_sd_pools=("lds_q25", "std"), n_pools=("pool_seed", "nunique"),
    )
    candidate_mean["lds_q25_sd_pools"] = candidate_mean.lds_q25_sd_pools.fillna(0)
    ranked, prioritized = rank_candidates(candidate_mean, traces, top_n)
    method_summary = method_by_pool.groupby("method", as_index=False).agg(
        mean_lds_q25=("mean_lds_q25", "mean"),
        sd_across_pools=("mean_lds_q25", "std"), n_pools=("pool_seed", "nunique"),
        n_targets=("n_targets", "first"),
    )
    method_summary["sd_across_pools"] = method_summary.sd_across_pools.fillna(0)
    return trajectories, method_by_pool, candidate_mean, method_summary, ranked, prioritized


def run(args: argparse.Namespace) -> None:
    if args.pairing_layer < 0 or args.optimal_layer < 0 or args.local_k < 1:
        raise ValueError("Layers must be nonnegative and local-k must be positive")
    if args.per_class < 1:
        raise ValueError("per-class must be >= 1")
    if args.top_n < 1:
        raise ValueError("top-n must be >= 1")
    seeds = parse_seeds(args.seeds)
    traces = read_trajectory_files(args.trajectories, args.property)
    pool = read_reference(args.pool, args.positive_label, args.negative_label)
    pool, diagnostics = clean_reference(pool, traces, args.exclude_csv)
    pos = pool[pool.label == args.positive_label].reset_index(drop=True)
    neg = pool[pool.label == args.negative_label].reset_index(drop=True)
    if min(len(pos), len(neg)) < args.per_class:
        raise ValueError(
            f"Need {args.per_class} independent proteins per label after exclusions; "
            f"have {len(pos)} positive and {len(neg)} negative"
        )
    diagnostics.update({"positive_after_exclusion": len(pos), "negative_after_exclusion": len(neg),
                        "per_class_per_pool": args.per_class, "seeds": seeds,
                        "local_k_requested": args.local_k})
    output_names = [
        "lds_step.csv", "lds_trajectory.csv", "lds_candidate_mean.csv",
        "lds_method_by_pool.csv", "lds_method_summary.csv", "pool_diagnostics.json",
        "ranked_candidates.csv", "prioritized_candidates.csv",
    ]
    outputs = [args.output_dir / name for name in output_names]
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("An output already exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embed = ESMCFeatures(args.model, args.device, args.pairing_layer, args.optimal_layer)
    step_frame, pool_rows = score_reference_pools(
        pool, pos, neg, traces, embed, seeds, args.per_class, args.local_k,
    )
    aggregate, method_by_pool, candidate_mean, method_summary, ranked, prioritized = (
        summarize_lds(step_frame, traces, args.top_n)
    )
    step_frame.to_csv(args.output_dir / "lds_step.csv", index=False)
    aggregate.to_csv(args.output_dir / "lds_trajectory.csv", index=False)
    candidate_mean.to_csv(args.output_dir / "lds_candidate_mean.csv", index=False)
    ranked.to_csv(args.output_dir / "ranked_candidates.csv", index=False)
    prioritized.to_csv(args.output_dir / "prioritized_candidates.csv", index=False)
    method_by_pool.to_csv(args.output_dir / "lds_method_by_pool.csv", index=False)
    method_summary.to_csv(args.output_dir / "lds_method_summary.csv", index=False)
    diagnostics["pool_details"] = pool_rows
    diagnostics["model"] = args.model
    diagnostics["property"] = args.property
    diagnostics["pairing_layer"] = args.pairing_layer
    diagnostics["optimal_layer"] = args.optimal_layer
    (args.output_dir / "pool_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print("Mean Q25 LDS across reference pools:\n", method_summary.to_string(index=False))
    print(f"Prioritized {len(prioritized)} of {len(ranked)} candidates by LDS")
    print("Saved:", args.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True, help="Independent CSV: sequence,label")
    parser.add_argument("--property", help="Property to select when inputs have a property column")
    parser.add_argument("--trajectories", type=Path, nargs="+", required=True,
                        help="One or more step-level CSVs or optimized-sequence CSVs with mutation_log")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, required=True,
                        help="Number of highest-LDS candidates to retain across input files")
    parser.add_argument("--pairing-layer", type=int, required=True, help="layer for matching backgrounds")
    parser.add_argument("--optimal-layer", type=int, required=True,
                        help="layer for sequence optimization before")
    parser.add_argument("--positive-label", default="1")
    parser.add_argument("--negative-label", default="0")
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--local-k", type=int, default=32)
    parser.add_argument("--seeds", default="17,29,43,71,93,100,101,2023,2068,3407")
    parser.add_argument("--exclude-csv", type=Path, action="append", default=[],
                        help="Optional construction/other excluded sequences CSV; repeat as needed")
    parser.add_argument("--model", default="esmc_600m")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
